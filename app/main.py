"""FastAPI 入口：接收让出方与划转量，查询配额、恢复状态与限速统计。"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import (
    RecoveryPendingError,
    TransferError,
    TransferService,
    WriteFailureError,
)
from .throttling import ThrottlingMonitor

log = logging.getLogger("quota-transfer")


def build_service_from_env() -> TransferService:
    crash_hook = None
    if os.environ.get("CRASH_AFTER_DONOR_WRITE") == "1":
        # 演示用：模拟“让出方降额后、接收方升额前进程崩溃退出”
        def crash_hook(_stage: str) -> None:
            os._exit(42)

    return TransferService(
        group_paths={
            "groupA": Path(os.environ.get("GROUP_A_CPU_MAX", "/cgroup/groupA/cpu.max")),
            "groupB": Path(os.environ.get("GROUP_B_CPU_MAX", "/cgroup/groupB/cpu.max")),
        },
        state_dir=Path(os.environ.get("STATE_DIR", "/state")),
        min_quota=int(os.environ.get("MIN_QUOTA", "20000")),
        crash_hook=crash_hook,
    )


def build_monitor_from_env(service: TransferService) -> ThrottlingMonitor:
    interval = float(os.environ.get("CPU_STAT_SAMPLE_INTERVAL", "1.0"))
    # cpu.stat 与 cpu.max 同目录（真实 cgroup 即如此）；可用
    # GROUP_A_CPU_STAT / GROUP_B_CPU_STAT 单独覆盖。
    stat_paths = {}
    for group, max_path in service.group_paths.items():
        env_key = f"{_env_prefix(group)}_CPU_STAT"
        stat_paths[group] = Path(
            os.environ.get(env_key, max_path.with_name("cpu.stat"))
        )
    return ThrottlingMonitor(
        service.group_paths,
        cpu_stat_paths=stat_paths,
        write_log=service.write_log,
        interval_seconds=interval,
    )


def _env_prefix(group: str) -> str:
    """groupA -> GROUP_A；groupB -> GROUP_B。"""
    if len(group) > 5 and group.startswith("group"):
        return f"GROUP_{group[5:].upper()}"
    return group.upper()


class TransferRequest(BaseModel):
    donor: str = Field(description="让出方组名（groupA 或 groupB）")
    amount: int = Field(gt=0, description="划转量（cpu.max quota 单位，微秒/周期）")


class TransferResponse(BaseModel):
    action: str
    result: str
    transfer_id: str
    donor: str
    receiver: str
    amount: int
    post: dict
    at: str


def create_app(
    service: TransferService, monitor: ThrottlingMonitor | None = None
) -> FastAPI:
    monitor = monitor or ThrottlingMonitor(
        service.group_paths, write_log=service.write_log
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动时先恢复：回退未登记成功的转移，保留已登记成功的新配额。
        # 恢复完成前（含恢复失败），recovered=False，新转移一律被拒绝。
        try:
            result = service.recover()
            log.info("startup recovery: %s", result)
        except Exception:
            log.exception("启动恢复失败；恢复完成前将拒绝新转移")
        monitor.start()
        try:
            yield
        finally:
            monitor.stop()

    app = FastAPI(title="CPU Quota Transfer Service", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/quotas")
    def get_quotas() -> dict:
        """查询两组当前配额。"""
        return service.quotas()

    @app.get("/status")
    def get_status() -> dict:
        """查询恢复状态与最近一次处理结果。"""
        return service.status()

    @app.get("/throttling")
    def get_throttling() -> dict:
        """查询两组按 cpu.max 档位累计的限速统计（仅本次启动）。"""
        return monitor.summary.snapshot_all()

    @app.get("/throttling/{group}")
    def get_throttling_group(group: str) -> dict:
        """按执行器组查询限速统计。"""
        try:
            return monitor.summary.snapshot(group)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"未知组 {group!r}，可选: {sorted(service.group_paths)}",
            )

    @app.post("/transfers", response_model=TransferResponse)
    def post_transfer(req: TransferRequest) -> dict:
        """发起一笔划转：donor 组让出 amount 给另一组。"""
        try:
            return service.transfer(req.donor, req.amount)
        except RecoveryPendingError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except TransferError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except WriteFailureError as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def app_factory() -> FastAPI:
    """uvicorn --factory 入口：延迟到真正启动时才读取环境、触碰文件系统。"""
    service = build_service_from_env()
    return create_app(service, build_monitor_from_env(service))
