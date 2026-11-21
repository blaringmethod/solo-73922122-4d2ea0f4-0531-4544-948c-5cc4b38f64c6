"""FastAPI 接口测试：上传、示例载入、判定、错误处理全流程。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import SAMPLE_FILES, app

client = TestClient(app)

DEMO = (Path(__file__).resolve().parent.parent / "samples" / "demo_monthly_reports.csv").read_text(
    encoding="utf-8"
)


def upload_csv(text: str, filename: str = "r.csv"):
    return client.post(
        "/api/upload",
        files={"file": (filename, text.encode("utf-8"), "text/csv")},
    )


def test_health_and_index():
    assert client.get("/api/health").json() == {"status": "ok"}
    resp = client.get("/")
    assert resp.status_code == 200
    assert "公交客流" in resp.text


def test_upload_and_analyze_rise_flow():
    resp = upload_csv(DEMO)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["record_count"] == 11
    assert len(payload["versions"]) == 3

    result = client.post(
        "/api/analyze",
        json={
            "dataset_id": payload["dataset_id"],
            "versions": ["V2020", "V2021", "V2022"],
            "start_month": "2020-07",
            "end_month": "2022-07",
        },
    ).json()
    assert result["verdict"] == "rise"
    assert result["verdict_label"] == "必涨"
    assert len(result["chain"]) >= 2
    assert result["product_low"] > 1


def test_upload_bad_csv_returns_422_with_errors():
    resp = upload_csv("foo,bar\n1,2\n")
    assert resp.status_code == 422
    assert "errors" in resp.json()


def test_sample_endpoints_cover_all_verdicts():
    # 矛盾示例
    key = "conflict"
    ds = client.post(f"/api/samples/{key}/load").json()
    result = client.post(
        "/api/analyze",
        json={
            "dataset_id": ds["dataset_id"],
            "versions": ["VA", "VB"],
            "start_month": "2023-01",
            "end_month": "2023-06",
        },
    ).json()
    assert result["verdict"] == "conflict"
    assert len(result["conflicts"]) >= 2

    # 缺依据示例
    ds2 = client.post("/api/samples/disconnected/load").json()
    result2 = client.post(
        "/api/analyze",
        json={
            "dataset_id": ds2["dataset_id"],
            "versions": ["VX", "VY"],
            "start_month": "2024-06",
            "end_month": "2025-06",
        },
    ).json()
    assert result2["verdict"] == "no_basis"


def test_unknown_dataset_id_404():
    resp = client.post(
        "/api/analyze",
        json={
            "dataset_id": "nope",
            "versions": ["V1"],
            "start_month": "2020-01",
            "end_month": "2020-02",
        },
    )
    assert resp.status_code == 404


def test_empty_version_selection_rejected():
    payload = upload_csv(DEMO).json()
    resp = client.post(
        "/api/analyze",
        json={
            "dataset_id": payload["dataset_id"],
            "versions": [],
            "start_month": "2020-07",
            "end_month": "2022-07",
        },
    )
    assert resp.status_code == 422


def test_records_endpoint():
    payload = upload_csv(DEMO).json()
    resp = client.get(f"/api/datasets/{payload['dataset_id']}/records", params={"version": "V2022"})
    assert resp.status_code == 200
    assert all(r["version"] == "V2022" for r in resp.json()["records"])


def test_all_sample_files_present():
    for key, path in SAMPLE_FILES.items():
        assert path.exists(), f"示例 {key} 缺失：{path}"


SIM_CSV = """版号,基期月,统计月,公布指数
A,2020-01,2020-01,100
A,2020-01,2020-02,110
B,2020-02,2020-02,100
B,2020-02,2020-03,120
B,2020-02,2020-04,150
C,2020-03,2020-03,100
C,2020-03,2020-04,120
D,2020-04,2020-04,100
D,2020-04,2020-05,130
E,2020-03,2020-03,100
E,2020-03,2020-04,120
F,2020-04,2020-04,100
F,2020-04,2020-05,130
"""

# 有真正备用通路的样例：2020-03↔2020-04 由 E、G 双版覆盖
REDUNDANT_CSV = """版号,基期月,统计月,公布指数
H,2020-01,2020-01,100
H,2020-01,2020-02,110
I,2020-02,2020-02,100
I,2020-02,2020-01,91
J,2020-03,2020-03,100
J,2020-03,2020-04,100
K,2020-03,2020-03,100
K,2020-03,2020-04,101
"""


def sim_payload():
    payload = upload_csv(SIM_CSV, "sim.csv").json()
    return payload["dataset_id"], payload


def redundant_payload():
    payload = upload_csv(REDUNDANT_CSV, "redundant.csv").json()
    return payload["dataset_id"], payload


def test_dataset_state_exposes_retired_and_protected():
    ds_id, payload = sim_payload()
    assert payload["active_versions"] == ["A", "B", "C", "D", "E", "F"]
    assert payload["retired_versions"] == []
    assert payload["protected_pairs"] == []
    state = client.get(f"/api/datasets/{ds_id}").json()
    assert all(v["active"] for v in state["versions"])


def test_simulate_reports_single_version_bridges():
    ds_id, _ = sim_payload()
    # 4↔5 原本由 D、F 双版覆盖；下架 F 后只剩 D，D 缺失即断
    sim = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": ["F"], "pairs": [{"start_month": "2020-04", "end_month": "2020-05"}]},
    ).json()
    check = sim["checks"][0]
    assert check["connected"] is True
    assert check["disconnected_immediately"] is False
    assert check["bridge_versions"] == ["D"]
    assert set(check["alternate_paths"]) == {"A", "B", "C", "E"}
    assert sim["eligible"] is False
    assert any("D" in r for r in sim["reasons"])


def test_simulate_and_confirm_flow_with_backup_paths():
    ds_id, _ = redundant_payload()
    sim = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={
            "retire_versions": ["J"],
            "pairs": [{"start_month": "2020-03", "end_month": "2020-04"}],
        },
    ).json()
    # J 下架后仍有 K，但 K 是唯一支撑 -> 不可确认
    assert sim["checks"][0]["bridge_versions"] == ["K"]
    assert sim["eligible"] is False

    # 下架 J、K 会立即断开
    sim2 = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={
            "retire_versions": ["J", "K"],
            "pairs": [{"start_month": "2020-03", "end_month": "2020-04"}],
        },
    ).json()
    assert sim2["checks"][0]["disconnected_immediately"] is True
    assert sim2["eligible"] is False

    # 下架 K，保护 1↔2（H、I 双版本）可确认
    sim3 = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={
            "retire_versions": ["K"],
            "pairs": [{"start_month": "2020-01", "end_month": "2020-02"}],
        },
    ).json()
    assert sim3["eligible"] is True
    assert sim3["checks"][0]["bridge_versions"] == []
    assert set(sim3["checks"][0]["alternate_paths"]) == {"H", "I", "J"}

    resp = client.post(
        f"/api/datasets/{ds_id}/confirm",
        json={"simulation_id": sim3["simulation_id"]},
    )
    assert resp.status_code == 200
    state = resp.json()
    assert state["retired_versions"] == ["K"]
    assert state["active_versions"] == ["H", "I", "J"]
    assert state["protected_pairs"] == [{"start_month": "2020-01", "end_month": "2020-02"}]

    result = client.post(
        "/api/analyze",
        json={
            "dataset_id": ds_id,
            "versions": state["active_versions"],
            "start_month": "2020-01",
            "end_month": "2020-02",
        },
    ).json()
    assert result["verdict"] != "no_basis"


def test_simulate_only_pairs_without_retiring():
    ds_id, _ = redundant_payload()
    sim = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": [], "pairs": [{"start_month": "2020-04", "end_month": "2020-03"}]},
    ).json()
    assert sim["eligible"] is True
    # 月对无方向，返回时按月份升序
    assert sim["checks"][0]["start_month"] == "2020-03"
    assert sim["checks"][0]["end_month"] == "2020-04"
    assert sim["draft_pairs"] == [{"start_month": "2020-03", "end_month": "2020-04"}]
    state = client.get(f"/api/datasets/{ds_id}").json()
    assert state["protected_pairs"] == []
    assert state["retired_versions"] == []


def test_simulate_disconnected_plan_not_confirmable():
    ds_id, _ = sim_payload()
    sim = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": ["D", "F"], "pairs": [{"start_month": "2020-03", "end_month": "2020-05"}]},
    ).json()
    assert sim["eligible"] is False
    assert sim["checks"][0]["disconnected_immediately"] is True
    resp = client.post(
        f"/api/datasets/{ds_id}/confirm",
        json={"simulation_id": sim["simulation_id"]},
    )
    assert resp.status_code == 409
    state = client.get(f"/api/datasets/{ds_id}").json()
    assert state["retired_versions"] == []
    assert state["protected_pairs"] == []


def test_simulate_validation_errors():
    ds_id, _ = sim_payload()
    assert client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": ["NOPE"], "pairs": []},
    ).status_code == 422
    same = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": [], "pairs": [{"start_month": "2020-01", "end_month": "2020-01"}]},
    )
    assert same.status_code == 422
    bad = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": [], "pairs": [{"start_month": "2020/01", "end_month": "2020-02"}]},
    )
    assert bad.status_code == 422


def test_confirm_twice_second_attempt_conflicts():
    ds_id, _ = redundant_payload()
    sim = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": ["K"], "pairs": [{"start_month": "2020-01", "end_month": "2020-02"}]},
    ).json()
    first = client.post(
        f"/api/datasets/{ds_id}/confirm",
        json={"simulation_id": sim["simulation_id"]},
    )
    assert first.status_code == 200
    # 同一试算快照不得重复确认
    second = client.post(
        f"/api/datasets/{ds_id}/confirm",
        json={"simulation_id": sim["simulation_id"]},
    )
    assert second.status_code == 404


def test_confirm_rejects_stale_simulation_after_other_tab_change():
    ds_id, _ = redundant_payload()
    sim = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": ["K"], "pairs": [{"start_month": "2020-01", "end_month": "2020-02"}]},
    ).json()
    # 模拟另一个标签页先提交了一个不同方案
    other = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": ["J"], "pairs": []},
    ).json()
    assert client.post(
        f"/api/datasets/{ds_id}/confirm",
        json={"simulation_id": other["simulation_id"]},
    ).status_code == 200
    stale = client.post(
        f"/api/datasets/{ds_id}/confirm",
        json={"simulation_id": sim["simulation_id"]},
    )
    assert stale.status_code == 409
    assert "重新试算" in stale.json()["detail"]["message"]
    state = client.get(f"/api/datasets/{ds_id}").json()
    # 失败时不产生部分写入
    assert state["retired_versions"] == ["J"]
    assert state["protected_pairs"] == []
    assert state["active_versions"] == ["H", "I", "K"]


def test_analyze_rejects_retired_version_but_records_remain():
    ds_id, _ = redundant_payload()
    sim = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": ["K"], "pairs": []},
    ).json()
    client.post(f"/api/datasets/{ds_id}/confirm", json={"simulation_id": sim["simulation_id"]})

    refused = client.post(
        "/api/analyze",
        json={
            "dataset_id": ds_id,
            "versions": ["K", "J"],
            "start_month": "2020-03",
            "end_month": "2020-04",
        },
    )
    assert refused.status_code == 422
    assert "已下架" in refused.json()["detail"]

    # 源行仍可查询
    records = client.get(f"/api/datasets/{ds_id}/records", params={"version": "K"}).json()
    assert records["records"] and all(r["version"] == "K" for r in records["records"])


def test_confirm_unknown_simulation_404():
    ds_id, _ = sim_payload()
    resp = client.post(f"/api/datasets/{ds_id}/confirm", json={"simulation_id": "nope"})
    assert resp.status_code == 404


def test_concurrent_confirms_only_one_wins():
    import threading

    ds_id, _ = redundant_payload()
    sims = []
    for retire, pair in (
        ("K", {"start_month": "2020-01", "end_month": "2020-02"}),
        ("J", {"start_month": "2020-03", "end_month": "2020-04"}),
    ):
        sims.append(client.post(
            f"/api/datasets/{ds_id}/simulate",
            json={"retire_versions": [retire], "pairs": [pair]},
        ).json())
    statuses = []
    lock = threading.Lock()

    def confirm(sim):
        resp = client.post(
            f"/api/datasets/{ds_id}/confirm",
            json={"simulation_id": sim["simulation_id"]},
        )
        with lock:
            statuses.append(resp.status_code)

    threads = [threading.Thread(target=confirm, args=(s,)) for s in sims]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(statuses) == [200, 409]
    state = client.get(f"/api/datasets/{ds_id}").json()
    assert state["revision"] == 1
    assert len(state["protected_pairs"]) == 1
    assert len(state["retired_versions"]) == 1


def test_dataset_state_after_confirm_reflects_retired_flag():
    ds_id, _ = redundant_payload()
    sim = client.post(
        f"/api/datasets/{ds_id}/simulate",
        json={"retire_versions": ["K"], "pairs": [{"start_month": "2020-01", "end_month": "2020-02"}]},
    ).json()
    client.post(f"/api/datasets/{ds_id}/confirm", json={"simulation_id": sim["simulation_id"]})
    state = client.get(f"/api/datasets/{ds_id}").json()
    flags = {v["version"]: v["active"] for v in state["versions"]}
    assert flags == {"H": True, "I": True, "J": True, "K": False}
