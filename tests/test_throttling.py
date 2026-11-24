"""cpu.stat 后台采样与按配额档位分档的限速统计测试。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.service as service_mod  # noqa: F401  （保持与其他测试一致的导入风格）
from app.cgroups import read_cpu_max
from app.main import create_app
from app.service import TransferService
from app.throttling import (
    CpuStat,
    CpuStatError,
    ThrottlingMonitor,
    ThrottlingSummary,
    WriteLog,
    parse_cpu_stat,
)

A, B = "groupA", "groupB"
PERIOD = 100000
MIN_Q = 20000
INIT_A, INIT_B = 100000, 50000


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def env(tmp_path):
    cg = tmp_path / "cgroup"
    for name, q in ((A, INIT_A), (B, INIT_B)):
        p = cg / name / "cpu.max"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{q} {PERIOD}\n")
        (cg / name / "cpu.stat").write_text("nr_periods 0\nnr_throttled 0\n")
    return cg, tmp_path / "state"


def make_monitor(env, clock=None, reader=None):
    cg, _ = env
    clock = clock or FakeClock()
    write_log = WriteLog(clock=clock)
    monitor = ThrottlingMonitor(
        {A: cg / A / "cpu.max", B: cg / B / "cpu.max"},
        cpu_stat_paths={A: cg / A / "cpu.stat", B: cg / B / "cpu.stat"},
        write_log=write_log,
        interval_seconds=1.0,
        reader=reader or read_stat,
        clock=clock,
    )
    return monitor, write_log, clock


STATIC_B = CpuStat(0, 0)


def read_stat(path: Path) -> CpuStat:
    text = Path(path).read_text()
    return parse_cpu_stat(text)


def set_stat(path: Path, periods: int, throttled: int) -> None:
    Path(path).write_text(
        f"usage_usec 1\nnr_periods {periods}\nnr_throttled {throttled}\n"
        "throttled_usec 1\n"
    )


class ScriptedReader:
    """按脚本顺序返回 cpu.stat，用来模拟采样值变化。

    ``default`` 用于脚本未覆盖的组（例如 groupB 保持静止）。
    """

    def __init__(self, stats, default: CpuStat | None = CpuStat(0, 0)):
        if isinstance(stats, dict):
            self.stats = {g: list(v) for g, v in stats.items()}
        else:
            self.stats = {A: list(stats)}
        self.default = default

    def __call__(self, path):
        group = Path(path).parent.name
        queue = self.stats.get(group)
        if not queue:
            if self.default is not None:
                return self.default
            raise CpuStatError("脚本已用完")
        return queue.pop(0)


# ---------------------------------------------------------------- 解析


def test_parse_cpu_stat_ignores_irrelevant_fields():
    stat = parse_cpu_stat(
        "usage_usec 1\nuser_usec 2\nsystem_usec 3\n"
        "nr_periods 10\nnr_throttled 4\nthrottled_usec 5\n"
    )
    assert stat == CpuStat(nr_periods=10, nr_throttled=4)


@pytest.mark.parametrize(
    "text",
    ["nr_periods 1\n", "nr_throttled 1\n", "nr_periods -1\nnr_throttled 0\n"],
)
def test_parse_cpu_stat_rejects_bad_input(text):
    with pytest.raises(CpuStatError):
        parse_cpu_stat(text)


# ---------------------------------------------------------------- 累计与分档


def test_deltas_accumulate_and_ratio(env):
    cg, _ = env
    set_stat(cg / A / "cpu.stat", 1000, 100)
    monitor, _, clock = make_monitor(env)
    monitor.sample_once()  # 基线，不计增量

    set_stat(cg / A / "cpu.stat", 1100, 130)
    clock.advance(1)
    monitor.sample_once()

    snap = monitor.summary.snapshot(A)
    assert snap["buckets"] == [
        {
            "quota": str(INIT_A),
            "period": PERIOD,
            "nr_periods": 100,
            "nr_throttled": 30,
            "throttled_ratio": 0.3,
        }
    ]
    # B 未变化：没有增量，也没有占比字段
    snap_b = monitor.summary.snapshot(B)
    assert snap_b["buckets"] == []
    assert "throttled_ratio" not in snap_b["totals"]
    assert snap_b["totals"]["nr_periods"] == 0


def test_bucket_by_quota_at_sampling_time(env):
    cg, _ = env
    a_script = iter(
        [
            (1000, 100),  # 基线，配额 100000
            (1100, 120),  # 配额 100000
            (1200, 150),  # 配额已变为 120000
        ]
    )
    monitor, _, clock = make_monitor(env)

    def reader(path):
        if Path(path).parent.name != A:
            return CpuStat(0, 0)
        periods, throttled = next(a_script)
        return CpuStat(periods, throttled)

    monitor._reader = reader
    monitor.sample_once()
    clock.advance(1)
    monitor.sample_once()
    (cg / A / "cpu.max").write_text(f"120000 {PERIOD}\n")
    clock.advance(1)
    monitor.sample_once()

    buckets = monitor.summary.snapshot(A)["buckets"]
    assert [(b["quota"], b["nr_periods"], b["nr_throttled"]) for b in buckets] == [
        ("100000", 100, 20),
        ("120000", 100, 30),
    ]
    assert buckets[0]["throttled_ratio"] == pytest.approx(0.2)
    assert buckets[1]["throttled_ratio"] == pytest.approx(0.3)


# ---------------------------------------------------------------- 计数器回退


def test_counter_regression_discards_delta_and_restarts(env):
    cg, _ = env
    set_stat(cg / A / "cpu.stat", 0, 0)
    stats = [
        CpuStat(1000, 100),
        CpuStat(1100, 120),  # +100/+20 正常累计
        CpuStat(1050, 110),  # nr_periods 回退：整段丢弃
        CpuStat(1150, 140),  # 从 1050 起算：+100/+30
    ]
    monitor, _, clock = make_monitor(env, reader=ScriptedReader(stats))
    for _ in stats:
        monitor.sample_once()
        clock.advance(1)

    snap = monitor.summary.snapshot(A)
    assert snap["buckets"][0]["nr_periods"] == 200
    assert snap["buckets"][0]["nr_throttled"] == 50
    assert snap["unattributable"]["nr_periods"] == 0


def test_throttled_counter_regression_also_discards(env):
    stats = [
        CpuStat(1000, 100),
        CpuStat(1100, 90),  # nr_throttled 回退：丢弃
        CpuStat(1200, 130),  # +100/+40
    ]
    monitor, _, clock = make_monitor(env, reader=ScriptedReader(stats))
    for _ in stats:
        monitor.sample_once()
        clock.advance(1)
    snap = monitor.summary.snapshot(A)
    assert snap["buckets"][0]["nr_periods"] == 100
    assert snap["buckets"][0]["nr_throttled"] == 40


# ---------------------------------------------------------------- 写入重叠归属


def test_write_overlap_makes_periods_unattributable(env):
    cg, _ = env
    monitor, write_log, clock = make_monitor(env)
    stats = ScriptedReader([CpuStat(1000, 100), CpuStat(1100, 150)])
    monitor._reader = stats
    monitor.sample_once()
    clock.advance(1)
    # 采样窗口内本组 cpu.max 被写（模拟划转）
    window = write_log.begin(A)
    write_log.end(window)
    monitor.sample_once()

    snap = monitor.summary.snapshot(A)
    assert snap["buckets"] == []
    assert snap["unattributable"]["nr_periods"] == 100
    assert snap["unattributable"]["nr_throttled"] == 50
    assert snap["unattributable"]["throttled_ratio"] == pytest.approx(0.5)
    # 另一组不受影响：B 无写入、且没有增量
    assert monitor.summary.snapshot(B)["unattributable"]["nr_periods"] == 0


def test_other_group_write_does_not_affect_this_group(env):
    monitor, write_log, clock = make_monitor(env)
    monitor._reader = ScriptedReader([CpuStat(1000, 100), CpuStat(1100, 120)])
    monitor.sample_once()
    clock.advance(1)
    window = write_log.begin(B)  # 只有 groupB 写入
    write_log.end(window)
    monitor.sample_once()

    snap = monitor.summary.snapshot(A)
    assert snap["buckets"][0]["nr_periods"] == 100
    assert snap["unattributable"]["nr_periods"] == 0


def test_write_restoring_original_quota_still_unattributable(env):
    """配额先降后写回原值：两次写入都在窗口内，增量仍无法归属。"""
    cg, _ = env
    monitor, write_log, clock = make_monitor(env)
    monitor._reader = ScriptedReader([CpuStat(0, 0), CpuStat(50, 10)])
    monitor.sample_once()
    clock.advance(0.5)
    path = cg / A / "cpu.max"
    original = path.read_text()
    for value in (30000, INIT_A):  # 降额，再恢复原值
        window = write_log.begin(A)
        path.write_text(f"{value} {PERIOD}\n")
        write_log.end(window)
        clock.advance(0.1)
    monitor.sample_once()

    snap = monitor.summary.snapshot(A)
    assert snap["buckets"] == []
    assert snap["unattributable"]["nr_periods"] == 50
    assert path.read_text() == original


def test_write_before_previous_sample_does_not_taint(env):
    """写入发生在上一段采样窗口之前，不影响本段归属。"""
    monitor, write_log, clock = make_monitor(env)
    monitor._reader = ScriptedReader([CpuStat(0, 0), CpuStat(100, 10)])
    window = write_log.begin(A)
    write_log.end(window)
    clock.advance(1)
    monitor.sample_once()  # 基线
    clock.advance(1)
    monitor.sample_once()
    snap = monitor.summary.snapshot(A)
    assert snap["buckets"][0]["nr_periods"] == 100
    assert snap["unattributable"]["nr_periods"] == 0


# ---------------------------------------------------------------- 容错 / 生命周期


def test_stat_read_failure_skips_sample_and_recovers(env):
    monitor, _, clock = make_monitor(env)
    monitor._reader = ScriptedReader([CpuStat(1000, 100)])
    monitor.sample_once()  # 基线

    def failing(_path):
        raise OSError("cpu.stat 暂时不可读")

    monitor._reader = failing
    clock.advance(1)
    monitor.sample_once()  # 不应抛错

    monitor._reader = ScriptedReader([CpuStat(1100, 120)])
    clock.advance(1)
    monitor.sample_once()  # 恢复后按上次基线累计
    assert monitor.summary.snapshot(A)["buckets"][0]["nr_periods"] == 100


def test_slow_stat_read_does_not_block_transfer(env):
    cg, state = env
    gate = threading.Event()
    entered = threading.Event()
    real_reader = read_stat

    def slow_reader(path: Path) -> CpuStat:
        entered.set()
        assert gate.wait(timeout=10), "测试未放行"
        return real_reader(path)

    svc = TransferService(
        {A: cg / A / "cpu.max", B: cg / B / "cpu.max"}, state, MIN_Q
    )
    monitor = ThrottlingMonitor(
        svc.group_paths,
        write_log=svc.write_log,
        interval_seconds=0.01,
        reader=slow_reader,
    )
    svc.recover()
    monitor.start()
    try:
        assert entered.wait(timeout=10), "后台采样未开始"
        started = time.monotonic()
        res = svc.transfer(B, 20000)
        elapsed = time.monotonic() - started
        assert res["result"] == "success"
        assert elapsed < 5, "cpu.stat 读取变慢不应阻塞划转"
        assert read_cpu_max(svc.group_paths[A]).quota_value == 120000
        assert read_cpu_max(svc.group_paths[B]).quota_value == 30000
    finally:
        gate.set()
        monitor.stop()
        svc.close()


def test_summary_only_covers_this_startup(env):
    monitor, _, clock = make_monitor(env)
    monitor._reader = ScriptedReader([CpuStat(0, 0), CpuStat(100, 10)])
    monitor.sample_once()
    clock.advance(1)
    monitor.sample_once()
    assert monitor.summary.snapshot(A)["buckets"][0]["nr_periods"] == 100

    fresh, _, _ = make_monitor(env)
    snap = fresh.summary.snapshot(A)
    assert snap["buckets"] == []
    assert snap["totals"] == {"nr_periods": 0, "nr_throttled": 0}
    assert snap["samples"] == 0


def test_summary_zero_periods_omits_ratio():
    summary = ThrottlingSummary([A])
    snap = summary.snapshot(A)
    assert "throttled_ratio" not in snap["totals"]
    assert "throttled_ratio" not in snap["unattributable"]


# ---------------------------------------------------------------- API


def make_app(tmp_path):
    cg = tmp_path / "cgroup"
    state = tmp_path / "state"
    for name, q in ((A, INIT_A), (B, INIT_B)):
        p = cg / name / "cpu.max"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{q} {PERIOD}\n")
        (cg / name / "cpu.stat").write_text("nr_periods 0\nnr_throttled 0\n")
    svc = TransferService(
        {A: cg / A / "cpu.max", B: cg / B / "cpu.max"}, state, MIN_Q
    )
    monitor = ThrottlingMonitor(
        svc.group_paths,
        cpu_stat_paths={A: cg / A / "cpu.stat", B: cg / B / "cpu.stat"},
        write_log=svc.write_log,
        interval_seconds=0.05,
    )
    return create_app(svc, monitor), svc, monitor, cg


def test_api_throttling_endpoints(tmp_path):
    app, svc, monitor, cg = make_app(tmp_path)
    with TestClient(app) as c:
        # 先建立基线（值为 0），再依次写入两段增量，避免与首采样竞争
        deadline = time.monotonic() + 5
        while monitor.summary.snapshot(A)["samples"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        set_stat(cg / A / "cpu.stat", 100, 10)
        deadline = time.monotonic() + 5
        while monitor.summary.snapshot(A)["totals"]["nr_periods"] < 100:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        set_stat(cg / A / "cpu.stat", 300, 40)
        deadline = time.monotonic() + 5
        while monitor.summary.snapshot(A)["totals"]["nr_periods"] < 300:
            assert time.monotonic() < deadline
            time.sleep(0.02)

        r = c.get("/throttling")
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {A, B}
        bucket = body[A]["buckets"][0]
        assert bucket["nr_periods"] == 300
        assert bucket["nr_throttled"] == 40
        assert bucket["throttled_ratio"] == pytest.approx(40 / 300)

        r = c.get(f"/throttling/{A}")
        assert r.status_code == 200
        assert r.json()["group"] == A
        assert c.get("/throttling/groupC").status_code == 404
    svc.close()


def test_api_throttling_available_before_first_sample(tmp_path):
    app, svc, _, _ = make_app(tmp_path)
    with TestClient(app) as c:
        r = c.get(f"/throttling/{B}")
        assert r.status_code == 200
        body = r.json()
        assert body["buckets"] == []
        assert body["totals"]["nr_periods"] == 0
    svc.close()


# ---------------------------------------------------------------- 服务写入被记录


def test_service_transfer_records_write_windows(env):
    cg, state = env
    svc = TransferService(
        {A: cg / A / "cpu.max", B: cg / B / "cpu.max"}, state, MIN_Q
    )
    try:
        svc.recover()
        svc.transfer(B, 20000)
        now = time.monotonic()
        # 两次写入（B 降、A 升）都已在刚刚的采样窗口内登记
        assert svc.write_log.overlaps(B, now - 10, now + 1)
        assert svc.write_log.overlaps(A, now - 10, now + 1)
    finally:
        svc.close()


def test_rollback_write_windows_recorded_on_recovery(env, tmp_path):
    """重启回退的写入也登记窗口：three writes? 两组的回退写入都可见。"""
    cg, state = env
    svc = TransferService(
        {A: cg / A / "cpu.max", B: cg / B / "cpu.max"}, state, MIN_Q
    )
    svc.close()
    (cg / A / "cpu.max").write_text(f"120000 {PERIOD}\n")
    (cg / B / "cpu.max").write_text(f"30000 {PERIOD}\n")
    from app.journal import STATE_IN_PROGRESS, Journal

    Journal(state / "transfer_journal.json").write(
        {
            "state": STATE_IN_PROGRESS,
            "transfer_id": "t-rb",
            "pre": {
                A: {"quota": str(INIT_A), "period": PERIOD},
                B: {"quota": str(INIT_B), "period": PERIOD},
            },
        }
    )
    restarted = TransferService(
        {A: cg / A / "cpu.max", B: cg / B / "cpu.max"}, state, MIN_Q
    )
    try:
        restarted.recover()
        now = time.monotonic()
        assert restarted.write_log.overlaps(A, now - 10, now + 1)
        assert restarted.write_log.overlaps(B, now - 10, now + 1)
    finally:
        restarted.close()
