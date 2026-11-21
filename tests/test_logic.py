"""核心判定逻辑的单元测试，直接运行：python -m pytest -q"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.logic import (
    CSVParseError,
    VERDICT_CONFLICT,
    VERDICT_FALL,
    VERDICT_NO_BASIS,
    VERDICT_RISE,
    VERDICT_UNKNOWN,
    analyze,
    parse_csv,
)

DEMO_CSV = """版号,基期月,统计月,公布指数
V2020,2020-01,2020-01,100
V2020,2020-01,2020-07,120
V2020,2020-01,2021-01,108
V2021,2021-01,2021-01,100
V2021,2021-01,2020-07,111
V2021,2021-01,2020-12,100
V2021,2021-01,2021-07,95
V2021,2021-01,2022-01,90
V2022,2022-01,2022-01,100
V2022,2022-01,2021-07,106
V2022,2022-01,2022-07,130
"""


def load(text: str):
    return parse_csv(text)


def all_versions(ds) -> list[str]:
    return ds.versions


def test_parse_valid_csv_and_metadata():
    ds = load(DEMO_CSV)
    assert len(ds.records) == 11
    assert ds.versions == ["V2020", "V2021", "V2022"]
    assert ds.version_base["V2021"] == "2021-01"
    assert "2020-07" in ds.months and "2022-07" in ds.months


def test_parse_accepts_english_headers():
    text = """version,base_month,stat_month,published_index
V1,2020-01,2020-02,103
"""
    ds = load(text)
    assert ds.records[0].version == "V1"
    assert ds.records[0].published_index == 103


def test_parse_collects_all_errors():
    text = """版号,基期月,统计月,公布指数
V1,2020/01,2020-02,-5
,2020-13,2020-02,abc
V1,2020-02,2020-03,110
"""
    with pytest.raises(CSVParseError) as ei:
        load(text)
    joined = " | ".join(ei.value.errors)
    assert "第 2 行" in joined
    assert "第 3 行" in joined
    assert "正整数" in joined
    assert "YYYY-MM" in joined


def test_parse_reports_conflicting_base_month_within_version():
    text = """版号,基期月,统计月,公布指数
V1,2020-01,2020-02,110
V1,2020-03,2020-04,120
"""
    with pytest.raises(CSVParseError) as ei:
        load(text)
    assert any("基期月冲突" in e for e in ei.value.errors)


def test_parse_missing_column():
    with pytest.raises(CSVParseError) as ei:
        load("版号,基期月,统计月\nV1,2020-01,2020-02\n")
    assert "缺少必需列" in ei.value.errors[0]


def test_parse_duplicate_rows_warn_not_fail():
    text = "版号,基期月,统计月,公布指数\nV1,2020-01,2020-01,100\nV1,2020-01,2020-01,100\n"
    ds = load(text)
    assert len(ds.records) == 1
    assert ds.warnings and "重复" in ds.warnings[0]


def test_rise_across_three_base_versions():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2020-07", "2022-07")
    assert res.verdict == VERDICT_RISE
    assert res.chain, "必涨时必须给出证明链"
    # 链上每条步骤都来自真实源记录
    rows = {st.row_no for st in res.chain}
    assert rows.issubset({r.row_no for r in ds.records})
    # 月份必须首尾衔接
    assert res.chain[0].from_month == "2020-07"
    assert res.chain[-1].to_month == "2022-07"
    for a, b in zip(res.chain, res.chain[1:]):
        assert a.to_month == b.from_month
    # 连乘下界严格大于 1
    assert res.product_low > 1.0


def test_fall_reverse_direction():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2022-07", "2020-07")
    assert res.verdict == VERDICT_FALL
    assert res.chain
    assert res.chain[0].from_month == "2022-07"
    assert res.chain[-1].to_month == "2020-07"
    assert res.product_high < 1.0


def test_unknown_when_interval_contains_one():
    # 同一统计月 100 指数 => 比值区间 [0.995, 1.005] 包含 1
    text = """版号,基期月,统计月,公布指数
V1,2020-01,2020-01,100
V1,2020-01,2020-02,100
"""
    ds = load(text)
    res = analyze(ds, ["V1"], "2020-01", "2020-02")
    assert res.verdict == VERDICT_UNKNOWN
    assert res.ratio_low < 1.0 < res.ratio_high


def test_no_basis_when_disconnected():
    text = """版号,基期月,统计月,公布指数
VX,2024-01,2024-01,100
VX,2024-01,2024-06,110
VY,2025-01,2025-01,100
VY,2025-01,2025-06,120
"""
    ds = load(text)
    res = analyze(ds, ["VX", "VY"], "2024-06", "2025-06")
    assert res.verdict == VERDICT_NO_BASIS
    assert not res.chain


def test_no_basis_when_month_absent():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2020-07", "2030-01")
    assert res.verdict == VERDICT_NO_BASIS


def test_conflict_negative_cycle_lists_incompatible_records():
    # VA 称 F(06)/F(01) ∈ [1.995, 2.005]；VB 称 F(01)/F(06) ∈ [0.485, 0.495]
    # 即 F(06)/F(01) ∈ [1/0.495, 1/0.485] ≈ [2.020, 2.062]，与 VA 区间不相交
    text = """版号,基期月,统计月,公布指数
VA,2023-01,2023-01,100
VA,2023-01,2023-06,200
VB,2023-06,2023-06,100
VB,2023-06,2023-01,49
"""
    ds = load(text)
    res = analyze(ds, ["VA", "VB"], "2023-01", "2023-06")
    assert res.verdict == VERDICT_CONFLICT
    assert len(res.conflicts) >= 2
    # 负环必须同时包含两个版次的非平凡记录
    # （CSV 含表头：VA 指数 200 在第 3 行，VB 指数 49 在第 5 行）
    rows = {r.row_no for r in res.conflicts}
    assert {3, 5}.issubset(rows)
    assert {r.version for r in res.conflicts} == {"VA", "VB"}


def test_compatible_duplicate_measurement_not_conflict():
    # 两个版次对同一对月份给出相容区间，系统应可行，不得误报矛盾
    text = """版号,基期月,统计月,公布指数
VA,2023-01,2023-01,100
VA,2023-01,2023-06,150
VB,2023-06,2023-06,100
VB,2023-06,2023-01,67
"""
    ds = load(text)
    res = analyze(ds, ["VA", "VB"], "2023-01", "2023-06")
    assert res.verdict in (VERDICT_RISE, VERDICT_UNKNOWN)
    assert res.verdict != VERDICT_CONFLICT


def test_single_record_boundary_index():
    # 指数 200：比值区间 [1.995, 2.005]，必涨
    text = """版号,基期月,统计月,公布指数
V1,2020-01,2020-01,100
V1,2020-01,2020-02,200
"""
    ds = load(text)
    res = analyze(ds, ["V1"], "2020-01", "2020-02")
    assert res.verdict == VERDICT_RISE
    # 单条记录直接给出比值区间 [1.995, 2.005]
    assert res.product_low == pytest.approx(1.995)
    assert res.product_high == pytest.approx(2.005)
    # 反方向必跌，区间整体取倒数
    rev = analyze(ds, ["V1"], "2020-02", "2020-01")
    assert rev.verdict == VERDICT_FALL
    assert rev.product_low == pytest.approx(1.0 / 2.005)
    assert rev.product_high == pytest.approx(1.0 / 1.995)


def test_chain_products_match_bounds():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2020-07", "2022-07")
    lo = math.prod(st.ratio_low for st in res.chain)
    hi = math.prod(st.ratio_high for st in res.chain)
    assert res.product_low == pytest.approx(lo, rel=1e-9)
    assert res.product_high == pytest.approx(hi, rel=1e-9)


def test_version_selection_filters_records():
    ds = load(DEMO_CSV)
    # 只选 V2020：2020-07 与 2022-07 不连通
    res = analyze(ds, ["V2020"], "2020-07", "2022-07")
    assert res.verdict == VERDICT_NO_BASIS


def test_same_start_and_end_month_is_unknown_not_rise_or_fall():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2021-07", "2021-07")
    assert res.verdict == VERDICT_UNKNOWN
    assert res.ratio_low == pytest.approx(1.0)
    assert res.ratio_high == pytest.approx(1.0)
    assert res.chain == []




# --------------------------------------------------------------------------- #
# 下架试算
# --------------------------------------------------------------------------- #
#
# 目标：下架旧版后，即使余下任一版次整版源行临时取不到，保护月对也必须
# 仍能换基。因此“备用通路”= 下架后仍连通，且不存在“该版缺失即断开”的
# 在架版次。

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
"""

# 2020-03↔2020-04 由 J、K 双版覆盖；2020-01↔2020-02 由 H、I 双版覆盖
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


def sim_dataset():
    return parse_csv(SIM_CSV)


def redundant_dataset():
    return parse_csv(REDUNDANT_CSV)


def test_simulate_pair_is_connected_and_has_no_single_version_bridge():
    from app.logic import simulate_retirement

    ds = redundant_dataset()
    sim = simulate_retirement(ds, "ds", [], [("2020-03", "2020-04")])
    check = sim.checks[0]
    assert check.connected
    assert check.bridge_versions == []
    assert set(check.alternate_paths) == {"H", "I", "J", "K"}
    assert sim.eligible
    # 只填月对不改变数据集
    assert ds.retired_versions == set()
    assert ds.protected_pairs == []


def test_simulate_single_version_bridge_blocks_eligibility():
    from app.logic import simulate_retirement

    ds = redundant_dataset()
    sim = simulate_retirement(ds, "ds", ["J"], [("2020-03", "2020-04")])
    check = sim.checks[0]
    # 仍连通（K 还在），但 K 一旦临时取不到就断开 -> 无备用通路
    assert check.connected
    assert not check.disconnected_immediately
    assert check.bridge_versions == ["K"]
    assert not sim.eligible
    assert any("K" in reason for reason in sim.reasons)


def test_simulate_lists_immediately_disconnected_pairs():
    from app.logic import simulate_retirement

    ds = sim_dataset()
    sim = simulate_retirement(ds, "ds", ["D"], [("2020-03", "2020-05")])
    assert not sim.eligible
    check = sim.checks[0]
    assert check.disconnected_immediately
    assert not check.connected
    assert any("2020-03" in r for r in sim.reasons)


def test_simulate_reports_every_single_version_bridge():
    from app.logic import simulate_retirement

    ds = sim_dataset()
    sim = simulate_retirement(ds, "ds", ["C"], [("2020-03", "2020-05")])
    check = sim.checks[0]
    # C 下架后仍可经 B、D 与中间月份 2020-02 连通；
    # 但 B 或 D 任一版缺失都会使 3↔5 断开
    assert check.connected
    assert set(check.bridge_versions) == {"B", "D"}
    assert set(check.alternate_paths) == {"A"}
    assert "B" not in check.alternate_paths
    assert "D" not in check.alternate_paths
    assert check.path and check.path[0]["from_month"] == "2020-03"
    assert [h["version"] for h in check.path] == ["B", "B", "D"]
    assert not sim.eligible


def test_simulate_uses_intermediate_months():
    from app.logic import simulate_retirement

    ds = sim_dataset()
    sim = simulate_retirement(ds, "ds", ["C"], [("2020-03", "2020-04")])
    check = sim.checks[0]
    assert check.connected
    # 3↔4 原本有一条 C 记录；C 下架后借用中间月份 2020-02 与 B
    hops = [(h["from_month"], h["to_month"], h["version"]) for h in check.path]
    assert hops == [
        ("2020-03", "2020-02", "B"),
        ("2020-02", "2020-04", "B"),
    ]
    assert check.bridge_versions == ["B"]
    assert not sim.eligible


def test_simulate_includes_registered_pairs():
    from app.logic import simulate_retirement

    ds = sim_dataset()
    ds.protected_pairs = [("2020-01", "2020-02")]
    sim = simulate_retirement(ds, "ds", ["B"], [])
    assert len(sim.checks) == 1
    check = sim.checks[0]
    assert (check.start_month, check.end_month) == ("2020-01", "2020-02")
    assert check.source == "registered"
    # 该月对只靠版次 A，不依赖 B，仍然连通；但 A 是唯一桥接
    assert check.connected
    assert check.bridge_versions == ["A"]
    assert not sim.eligible


def test_simulate_rejects_unknown_or_already_retired_versions():
    from app.logic import simulate_retirement

    ds = sim_dataset()
    with pytest.raises(ValueError):
        simulate_retirement(ds, "ds", ["ZZZ"], [])
    ds.retired_versions.add("A")
    with pytest.raises(ValueError):
        simulate_retirement(ds, "ds", ["A"], [])


def test_simulate_pairs_are_normalized_and_deduplicated():
    from app.logic import simulate_retirement, normalize_pair

    ds = redundant_dataset()
    pairs = list(dict.fromkeys([
        normalize_pair("2020-04", "2020-03"),
        normalize_pair("2020-03", "2020-04"),
    ]))
    assert pairs == [("2020-03", "2020-04")]
    sim = simulate_retirement(ds, "ds", [], pairs)
    assert len(sim.checks) == 1
    assert (sim.checks[0].start_month, sim.checks[0].end_month) == ("2020-03", "2020-04")


def test_simulate_draft_pair_already_registered_is_not_duplicated():
    from app.logic import simulate_retirement

    ds = redundant_dataset()
    ds.protected_pairs = [("2020-03", "2020-04")]
    sim = simulate_retirement(ds, "ds", ["J"], [("2020-03", "2020-04")])
    assert len(sim.checks) == 1
    assert sim.draft_pairs == []
    assert sim.checks[0].source == "registered"


def test_normalize_pair_rejects_same_or_bad_month():
    from app.logic import normalize_pair

    with pytest.raises(ValueError):
        normalize_pair("2020-01", "2020-01")
    with pytest.raises(ValueError):
        normalize_pair("2020/01", "2020-02")
    assert normalize_pair("2020-04", "2020-03") == ("2020-03", "2020-04")


def test_confirm_registers_new_pairs_and_retires_versions():
    from app.logic import simulate_retirement, apply_simulation

    ds = redundant_dataset()
    sim = simulate_retirement(ds, "ds", ["K"], [("2020-01", "2020-02")])
    # K 下架不影响 1↔2（H、I 双版本），方案可确认
    assert sim.eligible
    apply_simulation(ds, sim)
    assert ds.retired_versions == {"K"}
    assert ds.protected_pairs == [("2020-01", "2020-02")]
    assert ds.revision == 1
    res = analyze(ds, ds.active_versions, "2020-01", "2020-02")
    assert res.verdict != VERDICT_NO_BASIS


def test_confirm_refuses_after_versions_change():
    from app.logic import simulate_retirement, apply_simulation, SimulationConflict

    ds = redundant_dataset()
    sim = simulate_retirement(ds, "ds", ["K"], [("2020-01", "2020-02")])
    ds.retired_versions.add("H")  # 另一个标签页先做了变更
    ds.revision += 1
    with pytest.raises(SimulationConflict) as ei:
        apply_simulation(ds, sim)
    assert "重新试算" in str(ei.value)
    # 失败时不产生部分写入
    assert ds.retired_versions == {"H"}
    assert ds.protected_pairs == []
    assert ds.revision == 1


def test_confirm_refuses_after_registered_pairs_change():
    from app.logic import simulate_retirement, apply_simulation, SimulationConflict

    ds = redundant_dataset()
    sim = simulate_retirement(ds, "ds", ["K"], [("2020-01", "2020-02")])
    ds.protected_pairs = [("2020-03", "2020-04")]
    ds.revision += 1
    with pytest.raises(SimulationConflict):
        apply_simulation(ds, sim)
    assert ds.retired_versions == set()
    assert ds.protected_pairs == [("2020-03", "2020-04")]


def test_confirm_rejects_ineligible_plan_without_changes():
    from app.logic import simulate_retirement, apply_simulation, SimulationConflict

    ds = sim_dataset()
    sim = simulate_retirement(ds, "ds", ["D"], [("2020-03", "2020-05")])
    assert not sim.eligible
    with pytest.raises(SimulationConflict):
        apply_simulation(ds, sim)
    assert ds.retired_versions == set()
    assert ds.protected_pairs == []
    assert ds.revision == 0


def test_retired_versions_excluded_from_analyze():
    ds = sim_dataset()
    ds.retired_versions.add("D")
    # 即使调用方坚持勾选已下架版次，判定也不使用其记录
    res = analyze(ds, all_versions(ds), "2020-04", "2020-05")
    assert res.verdict == VERDICT_NO_BASIS


def test_retirement_keeps_all_pairs_comparable_when_single_version_missing():
    from app.logic import simulate_retirement, apply_simulation

    ds = redundant_dataset()
    sim = simulate_retirement(
        ds, "ds", ["K"], [("2020-01", "2020-02"), ("2020-03", "2020-04")],
    )
    # 3↔4 只剩 J，J 一旦缺失即断 -> J 是单版桥接，方案不可确认
    assert not sim.eligible
    assert sim.checks[1].bridge_versions == ["J"]

    # 改为同时下架 J、K，3↔4 直接断开
    sim2 = simulate_retirement(
        ds, "ds", ["J", "K"], [("2020-03", "2020-04")],
    )
    assert not sim2.eligible
    assert sim2.checks[0].disconnected_immediately


def test_backup_sample_is_consistent_and_demo_ready():
    from app.logic import simulate_retirement, apply_simulation

    text = (Path(__file__).resolve().parent.parent / "samples"
            / "backup_paths_reports.csv").read_text(encoding="utf-8")
    ds = parse_csv(text)
    # 下架 V_OLD 后，保护月对仍各有至少两条独立通路
    sim = simulate_retirement(
        ds, "ds", ["V_OLD"],
        [("2020-07", "2022-07"), ("2021-07", "2022-07")],
    )
    assert sim.eligible
    for check in sim.checks:
        assert check.bridge_versions == []
    apply_simulation(ds, sim)
    # 下架后每个登记月对仍可判定
    for start, end in ds.protected_pairs:
        res = analyze(ds, ds.active_versions, start, end)
        assert res.verdict not in (VERDICT_NO_BASIS, VERDICT_CONFLICT)
        assert res.chain


def test_backup_sample_rejects_removing_both_backups_for_a_pair():
    from app.logic import simulate_retirement

    text = (Path(__file__).resolve().parent.parent / "samples"
            / "backup_paths_reports.csv").read_text(encoding="utf-8")
    ds = parse_csv(text)
    sim = simulate_retirement(ds, "ds", ["V_A", "V_C"], [("2020-07", "2022-07")])
    check = sim.checks[0]
    assert check.connected
    assert check.bridge_versions == ["V_B"]
    assert not sim.eligible
