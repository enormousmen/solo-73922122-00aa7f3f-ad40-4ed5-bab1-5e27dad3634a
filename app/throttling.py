"""cpu.stat 后台采样与限速（throttling）统计。

语义要点：

* 后台线程每隔 ``interval_seconds`` 秒尝试读取两组的 cpu.stat，用相邻两次
  成功读数的 ``nr_periods`` / ``nr_throttled`` 增量累计；
* 增量按“当时”的 ``cpu.max`` 配额与周期分档（bucket）；
* 某个计数器回退（本次读数小于上次）时，整段差值丢弃，并以本次读数重新起算；
* 每段采样的归属窗口是 [上次采样开始, 本次采样结束]。如果本组的 cpu.max
  在窗口内有写入（包括把配额写回原值的写入），这段增量一律计入
  ``unattributable``（无法归属），不进入任何配额档；另一组写入不影响本组；
* 统计只保存在内存中，每次进程启动重新开始，不落盘、不跨重启合并；
* 采样线程不持有划转服务的主锁，cpu.stat 读取再慢也不会阻塞划转。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .cgroups import CpuMax, read_cpu_max

log = logging.getLogger("quota-transfer.throttling")


class CpuStatError(Exception):
    """cpu.stat 内容非法（缺少计数或计数为负）。"""


@dataclass(frozen=True)
class CpuStat:
    nr_periods: int
    nr_throttled: int


def parse_cpu_stat(text: str) -> CpuStat:
    """解析 cgroup v2 cpu.stat，只关心 nr_periods / nr_throttled。

    cpu.stat 的其余字段（usage_usec、throttled_usec 等）以及未知字段都忽略。
    """
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        key, raw = parts
        try:
            values[key] = int(raw)
        except ValueError:
            continue
    try:
        nr_periods = values["nr_periods"]
        nr_throttled = values["nr_throttled"]
    except KeyError as e:
        raise CpuStatError(f"cpu.stat 缺少字段 {e.args[0]!r}") from e
    if nr_periods < 0 or nr_throttled < 0:
        raise CpuStatError("cpu.stat 计数不能为负")
    return CpuStat(nr_periods=nr_periods, nr_throttled=nr_throttled)


def read_cpu_stat(path: Path) -> CpuStat:
    with open(path, "r") as f:
        return parse_cpu_stat(f.read())


# ---------------------------------------------------------------------------
# cpu.max 写入窗口：用于判断采样区间是否与本组写入重叠
# ---------------------------------------------------------------------------


@dataclass
class _WriteWindow:
    group: str
    start: float
    end: float | None = None


class WriteLog:
    """记录每次 cpu.max 写入的起止单调时钟，供采样线程做区间重叠判断。"""

    def __init__(
        self,
        retention_seconds: float = 3600.0,
        max_completed: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._clock = clock
        self._retention = retention_seconds
        self._max_completed = max_completed
        self._lock = threading.Lock()
        self._windows: list[_WriteWindow] = []

    def begin(self, group: str) -> _WriteWindow:
        window = _WriteWindow(group=group, start=self._clock())
        with self._lock:
            self._prune(window.start)
            self._windows.append(window)
        return window

    def end(self, window: _WriteWindow) -> None:
        with self._lock:
            window.end = self._clock()

    def overlaps(self, group: str, start: float, end: float) -> bool:
        """返回 [start, end] 是否与 group 的任一 cpu.max 写入窗口重叠。"""
        with self._lock:
            return any(
                w.group == group
                and w.start <= end
                and (w.end is None or w.end >= start)
                for w in self._windows
            )

    def _prune(self, now: float) -> None:
        cutoff = now - self._retention
        self._windows = [
            w for w in self._windows if w.end is None or w.end >= cutoff
        ]
        completed = [w for w in self._windows if w.end is not None]
        if len(completed) > self._max_completed:
            open_windows = [w for w in self._windows if w.end is None]
            self._windows = completed[-self._max_completed :] + open_windows


# ---------------------------------------------------------------------------
# 汇总（仅内存，进程启动后从零开始）
# ---------------------------------------------------------------------------


@dataclass
class _Bucket:
    quota: str
    period: int
    nr_periods: int = 0
    nr_throttled: int = 0


@dataclass
class _GroupStats:
    buckets: dict[tuple[str, int], _Bucket] = field(default_factory=dict)
    unattr_periods: int = 0
    unattr_throttled: int = 0
    samples: int = 0
    last_sample_at: str | None = None
    # 无法归属的原因统计，按“写了 cpu.max” / “读不到 cpu.max”区分
    unattr_reasons: set[str] = field(default_factory=set)


def _ratio(nr_periods: int, nr_throttled: int) -> float | None:
    """限速周期占比；周期数为 0 时没有占比（返回 None，输出时省略该字段）。"""
    if nr_periods <= 0:
        return None
    return nr_throttled / nr_periods


class ThrottlingSummary:
    """按执行器组、按 cpu.max 档位累计限速周期。"""

    def __init__(self, groups: list[str] | None = None):
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._lock = threading.Lock()
        self._groups: dict[str, _GroupStats] = {
            g: _GroupStats() for g in (groups or [])
        }

    def record(
        self,
        group: str,
        previous: CpuStat | None,
        current: CpuStat,
        quota: CpuMax | None,
        overlapped: bool,
    ) -> None:
        """登记一次成功采样。

        previous 为 None 表示启动后的第一次采样，只建立基线。
        任一计数回退时丢弃本段差值，current 自动成为下次的基线。
        """
        with self._lock:
            stats = self._groups.setdefault(group, _GroupStats())
            stats.samples += 1
            stats.last_sample_at = datetime.now(timezone.utc).isoformat()
            if previous is None:
                return

            delta_periods = current.nr_periods - previous.nr_periods
            delta_throttled = current.nr_throttled - previous.nr_throttled
            if delta_periods < 0 or delta_throttled < 0:
                # 计数器回退：整段差值不可信，丢弃；下次从本次读数起算。
                return
            if delta_periods == 0 and delta_throttled == 0:
                # 无新增周期，不必登记空档（查询输出也会省略零周期的占比）。
                return
            if overlapped or quota is None:
                # 本组 cpu.max 在采样窗口内被写过（或采样时读不到配额）：
                # 这些周期无法安全归属到某个配额档。
                stats.unattr_periods += delta_periods
                stats.unattr_throttled += delta_throttled
                stats.unattr_reasons.add(
                    "write_overlap" if overlapped else "cpu_max_unreadable"
                )
                return

            key = (quota.quota, quota.period)
            bucket = stats.buckets.get(key)
            if bucket is None:
                bucket = _Bucket(quota=quota.quota, period=quota.period)
                stats.buckets[key] = bucket
            bucket.nr_periods += delta_periods
            bucket.nr_throttled += delta_throttled

    def _snapshot_group(self, group: str) -> dict:
        stats = self._groups[group]
        buckets = sorted(
            stats.buckets.values(), key=lambda b: (_quota_sort_key(b.quota), b.period)
        )
        bucket_items = []
        for b in buckets:
            item = {
                "quota": b.quota,
                "period": b.period,
                "nr_periods": b.nr_periods,
                "nr_throttled": b.nr_throttled,
            }
            ratio = _ratio(b.nr_periods, b.nr_throttled)
            if ratio is not None:
                item["throttled_ratio"] = ratio
            bucket_items.append(item)

        unattr = {
            "nr_periods": stats.unattr_periods,
            "nr_throttled": stats.unattr_throttled,
            "reasons": sorted(stats.unattr_reasons),
        }
        unattr_ratio = _ratio(stats.unattr_periods, stats.unattr_throttled)
        if unattr_ratio is not None:
            unattr["throttled_ratio"] = unattr_ratio

        total_periods = stats.unattr_periods + sum(
            b.nr_periods for b in stats.buckets.values()
        )
        total_throttled = stats.unattr_throttled + sum(
            b.nr_throttled for b in stats.buckets.values()
        )
        totals = {
            "nr_periods": total_periods,
            "nr_throttled": total_throttled,
        }
        total_ratio = _ratio(total_periods, total_throttled)
        if total_ratio is not None:
            totals["throttled_ratio"] = total_ratio

        return {
            "group": group,
            "started_at": self.started_at,
            "samples": stats.samples,
            "last_sample_at": stats.last_sample_at,
            "buckets": bucket_items,
            "unattributable": unattr,
            "totals": totals,
        }

    def snapshot(self, group: str) -> dict:
        with self._lock:
            if group not in self._groups:
                raise KeyError(group)
            return self._snapshot_group(group)

    def snapshot_all(self) -> dict:
        with self._lock:
            return {g: self._snapshot_group(g) for g in self._groups}


def _quota_sort_key(quota: str) -> float:
    try:
        return float(int(quota))
    except ValueError:
        # "max" 等非数值档位排在数值档位之后。
        return float("inf")


# ---------------------------------------------------------------------------
# 后台采样线程
# ---------------------------------------------------------------------------


class ThrottlingMonitor:
    """每秒读取两组 cpu.stat，累计限速统计；只读、不持有划转服务的主锁。"""

    def __init__(
        self,
        group_paths: dict[str, Path],
        cpu_stat_paths: dict[str, Path] | None = None,
        write_log: WriteLog | None = None,
        interval_seconds: float = 1.0,
        reader: Callable[[Path], CpuStat] = read_cpu_stat,
        quota_reader: Callable[[Path], CpuMax] = read_cpu_max,
        clock: Callable[[], float] = time.monotonic,
    ):
        if interval_seconds <= 0:
            raise ValueError("采样间隔必须为正数")
        self.group_paths = {g: Path(p) for g, p in group_paths.items()}
        self.cpu_stat_paths = {
            g: Path(p)
            for g, p in (
                cpu_stat_paths
                or {
                    g: p.with_name("cpu.stat")
                    for g, p in self.group_paths.items()
                }
            ).items()
        }
        missing = set(self.group_paths) - set(self.cpu_stat_paths)
        if missing:
            raise ValueError(f"缺少 cpu.stat 路径: {sorted(missing)}")
        self.write_log = write_log or WriteLog(clock=clock)
        self.interval_seconds = float(interval_seconds)
        self.summary = ThrottlingSummary(list(self.group_paths))
        self._reader = reader
        self._quota_reader = quota_reader
        self._clock = clock
        self._prev: dict[str, tuple[float, CpuStat]] = {}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._warned: set[str] = set()

    # -- 生命周期 ---------------------------------------------------------
    def start(self) -> None:
        if any(t.is_alive() for t in self._threads):
            return
        self._stop.clear()
        # 每组一个采样线程：某组 cpu.stat 读取变慢不会拖住另一组。
        self._threads = [
            threading.Thread(
                target=self._run_group,
                args=(group,),
                name=f"cpu-stat-sampler-{group}",
                daemon=True,
            )
            for group in self.group_paths
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        self._threads = []

    def _run_group(self, group: str) -> None:
        while not self._stop.is_set():
            started = self._clock()
            try:
                self._sample_group(group)
            except Exception:  # pragma: no cover - 防御：采样线程不能死
                log.exception("cpu.stat 采样异常（%s）", group)
            elapsed = self._clock() - started
            if self._stop.wait(max(0.0, self.interval_seconds - elapsed)):
                break

    # -- 采样 -------------------------------------------------------------
    def sample_once(self) -> None:
        for group in self.group_paths:
            if self._stop.is_set():
                return
            self._sample_group(group)

    def _sample_group(self, group: str) -> None:
        t_start = self._clock()
        # 先读配额、再读 cpu.stat：配额反映的是采样窗口内的档位；
        # 若窗口内（含读到 stat 之后、记录结束时间之前）本组有写入，
        # 该段增量会被判为无法归属，不会用错档位。
        try:
            quota = self._quota_reader(self.group_paths[group])
        except Exception as e:
            log.debug("读取 %s 失败: %s", self.group_paths[group], e)
            quota = None

        try:
            stat = self._reader(self.cpu_stat_paths[group])
        except Exception as e:
            if group not in self._warned:
                self._warned.add(group)
                log.warning("读取 %s 失败，稍后重试: %s", self.cpu_stat_paths[group], e)
            else:
                log.debug("读取 %s 失败: %s", self.cpu_stat_paths[group], e)
            return
        self._warned.discard(group)

        t_end = self._clock()
        previous = self._prev.get(group)
        window_start = previous[0] if previous is not None else t_start
        overlapped = self.write_log.overlaps(group, window_start, t_end)
        self.summary.record(
            group,
            previous[1] if previous is not None else None,
            stat,
            quota,
            overlapped,
        )
        self._prev[group] = (t_start, stat)
