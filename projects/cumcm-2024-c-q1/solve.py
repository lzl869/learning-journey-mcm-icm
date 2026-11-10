"""2024 国赛 C 题问题一。运行 python solve.py --help。"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent
YEARS = range(2024, 2031)
SEASONS = {"单季": 1, "第一季": 1, "第二季": 2}
WATER = "水浇地"
SMART = "智慧大棚"
GREEN = "普通大棚"
DRY = {"平旱地", "梯田", "山坡地"}
EPS = 0.00002


def read_data(directory):
    """用编号连接表。地块列的合并单元格需要沿用上一行地块。"""
    for name in ("附件1.xlsx", "附件2.xlsx"):
        if not (directory / name).is_file():
            raise SystemExit(f"缺少输入文件：{directory / name}")
    a = load_workbook(directory / "附件1.xlsx", data_only=True)
    b = load_workbook(directory / "附件2.xlsx", data_only=True)
    plots = {p.strip(): (kind.strip(), float(area)) for p, kind, area, *_ in
             list(a["乡村的现有耕地"].values)[1:] if isinstance(area, (int, float))}
    crops = {j: name.strip() for j, name, *_ in
             list(a["乡村种植的农作物"].values)[1:] if isinstance(j, int)}
    beans = {j for j, _, kind, *_ in list(a["乡村种植的农作物"].values)[1:]
             if isinstance(j, int) and "豆类" in kind}
    stats, prices = {}, defaultdict(set)
    for index, j, _, kind, season, yield_, cost, price, *_ in list(
        b["2023年统计的相关数据"].values
    )[1:]:
        if not isinstance(index, int):
            continue
        price = float(np.mean([float(v) for v in str(price).replace("–", "-").split("-")]))
        stats[kind.strip(), SEASONS[season.strip()], j] = (float(yield_), float(cost))
        prices[SEASONS[season.strip()], j].add(price)
    # 附件2注明智慧大棚第一季沿用普通大棚第一季参数。
    for (kind, season, j), values in list(stats.items()):
        if kind == GREEN and season == 1:
            stats[SMART, 1, j] = values
    if any(len(v) != 1 for v in prices.values()):
        raise ValueError("同作物出现不同价格，须先确定市场合并口径")
    price = {j: next(iter(v)) for j, v in prices.items()}
    history, demand = defaultdict(float), defaultdict(float)
    plot = None
    for p, j, name, _, area, season in list(b["2023年的农作物种植情况"].values)[1:]:
        if not isinstance(j, int):
            continue
        plot = p.strip() if p else plot
        s = SEASONS[season.strip()]
        kind = plots[plot][0]
        if crops[j] != name.strip() or float(area) <= 0:
            raise ValueError(f"历史种植记录有误：{plot}, {j}, {name}, {area}")
        history[plot, s, j] += float(area)
        demand[s, j] += float(area) * stats[kind, s, j][0]
    if len(plots) != 54 or len(crops) != 41 or {j for _, j in price} != set(crops):
        raise ValueError("地块或作物数据不完整")
    if not math.isclose(sum(a for k, a in plots.values() if k not in {GREEN, SMART}), 1201):
        raise ValueError("露天耕地面积与题面不符")
    if set(p for p, s, j in history) != set(plots):
        raise ValueError("2023 年种植记录缺少地块")
    return plots, crops, beans, stats, price, history, demand


class Model:
    """把每条线性约束存成稀疏矩阵的一行。"""
    def __init__(self):
        self.lower, self.upper, self.objective, self.integer = [], [], [], []
        self.rows, self.cols, self.coeffs, self.lo, self.hi = [], [], [], [], []

    def variable(self, upper, objective=0, binary=False):
        index = len(self.lower)
        self.lower.append(0)
        self.upper.append(upper)
        self.objective.append(objective)
        self.integer.append(int(binary))
        return index

    def row(self, terms, lo=-np.inf, hi=np.inf):
        index = len(self.lo)
        for col, coeff in terms:
            self.rows.append(index)
            self.cols.append(col)
            self.coeffs.append(coeff)
        self.lo.append(lo)
        self.hi.append(hi)

    def solve(self, seconds, gap):
        matrix = coo_matrix((self.coeffs, (self.rows, self.cols)),
                            shape=(len(self.lo), len(self.lower))).tocsc()
        result = milp(self.objective, integrality=self.integer,
                      bounds=Bounds(self.lower, self.upper),
                      constraints=LinearConstraint(matrix, self.lo, self.hi),
                      options={"time_limit": seconds, "mip_rel_gap": gap})
        if result.x is None:
            raise RuntimeError(f"求解器没有找到可行解：{result.message}")
        v = result.x
        lhs = matrix @ v
        violation = max(np.max(np.array(self.lo) - lhs), np.max(lhs - self.hi),
                        np.max(np.array(self.lower) - v), np.max(v - self.upper))
        binary = v[np.array(self.integer, dtype=bool)]
        integrality = np.max(np.abs(binary - np.rint(binary)))
        if violation > EPS or integrality > EPS:
            raise RuntimeError(f"约束或整数性检查失败：{violation}, {integrality}")
        return result, float(violation), float(integrality)


def build(data, discount, fraction, cap):
    plots, crops, beans, stats, price, history, demand = data
    m = Model()
    x, z = {}, {}
    by_plot, by_market = defaultdict(list), defaultdict(list)
    for t in YEARS:
        for p, (kind, area) in plots.items():
            for (k, s, j), (yield_, cost) in stats.items():
                if k != kind:
                    continue
                key = (t, p, s, j)
                # SciPy 最小化负利润。产量乘折扣价是基础收入，原价量再补差价。
                x[key] = m.variable(area, cost - discount * price[s, j] * yield_)
                z[key] = m.variable(1, binary=True)
                m.row([(x[key], 1), (z[key], -area)], hi=0)
                m.row([(x[key], 1), (z[key], -area * fraction)], lo=0)
                by_plot[t, p, s].append(key)
                by_market[t, s, j].append(key)
    sales = {}
    for (t, s, j), keys in by_market.items():
        sales[t, s, j] = m.variable(demand[s, j], -(1 - discount) * price[s, j])
        m.row([(sales[t, s, j], 1)] +
              [(x[key], -stats[plots[key[1]][0], s, j][0]) for key in keys], hi=0)
        m.row([(z[key], 1) for key in keys], hi=cap)

    for p, (kind, area) in plots.items():
        seasons = (1,) if kind in DRY else (1, 2)
        # 2023 年最后一茬接入 2024 年。稻作年水浇地的最后一茬在第一个槽位。
        last = max(s for q, s, j in history if q == p)
        for key in by_plot[2024, p, 1]:
            if history.get((p, last, key[3]), 0) > 0:
                m.row([(z[key], 1)], hi=0)
        for t in YEARS:
            for s in seasons:
                keys = by_plot[t, p, s]
                terms = [(x[key], 1) for key in keys]
                if kind == WATER and s == 2:
                    rice = z[t, p, 1, 16]
                    m.row(terms + [(rice, area)], lo=area, hi=area)
                    m.row([(z[key], 1) for key in keys] + [(rice, 1)], lo=1, hi=1)
                else:
                    m.row(terms, lo=area, hi=area)
                if kind == WATER and s == 1:
                    # 有水稻就整块种稻，第二季由上面的约束清空。
                    rice_key = (t, p, 1, 16)
                    m.row([(x[rice_key], 1), (z[rice_key], -area)], lo=0, hi=0)

                following = (t, p, 2) if s == 1 and len(seasons) == 2 else (t + 1, p, 1)
                for key in keys:
                    neighbor = (*following, key[3])
                    if neighbor in z:
                        m.row([(z[key], 1), (z[neighbor], 1)], hi=1)
                if kind == WATER and s == 1 and t < YEARS[-1]:
                    # 跳过稻作年的空白第二季，仍需检查下一年能否再种稻。
                    m.row([(z[t, p, 1, 16], 1), (z[t + 1, p, 1, 16], 1)], hi=1)

        for start in range(2023, 2029):
            past = sum(v for (q, s, j), v in history.items()
                       if q == p and j in beans) if start == 2023 else 0
            terms = [(x[key], 1) for t in range(max(start, 2024), start + 3)
                     for s in seasons for key in by_plot[t, p, s] if key[3] in beans]
            m.row(terms, lo=area - past)
    return m, x, sales


def verify(data, areas, discount, fraction, cap):
    """独立按种植面积重算规则、产量、销量和利润。"""
    plots, crops, beans, stats, price, history, demand = data
    plant = defaultdict(dict)
    market = defaultdict(float)
    counts = defaultdict(int)
    cost = defaultdict(float)
    for (t, p, s, j), area in areas.items():
        kind, size = plots[p]
        if (kind, s, j) not in stats or area + EPS < fraction * size:
            raise ValueError(f"不适种或面积过小：{t}, {p}, {s}, {j}")
        plant[t, p, s][j] = area
        yield_, unit_cost = stats[kind, s, j]
        market[t, s, j] += area * yield_
        counts[t, s, j] += 1
        cost[t] += area * unit_cost
    if max(counts.values()) > cap:
        raise ValueError("种植地块数超限")
    for p, (kind, size) in plots.items():
        past_seasons = sorted({s for q, s, j in history if q == p})
        previous = {j for q, s, j in history if q == p and s == past_seasons[-1]}
        bean_area = {2023: sum(v for (q, s, j), v in history.items() if q == p and j in beans)}
        for t in YEARS:
            first, second = plant[t, p, 1], plant[t, p, 2]
            one_season = kind in DRY or (kind == WATER and 16 in first)
            expected = (size, 0 if one_season else size)
            if any(abs(sum(batch.values()) - target) > EPS
                   for batch, target in zip((first, second), expected)):
                raise ValueError(f"季面积不符：{t}, {p}")
            if kind == WATER and ((16 in first and len(first) != 1)
                                  or (not one_season and len(second) != 1)):
                raise ValueError(f"水浇地模式错误：{t}, {p}")
            for batch in (first, second):
                if batch:
                    if previous.intersection(batch):
                        raise ValueError(f"连续重茬：{t}, {p}")
                    previous = set(batch)
            bean_area[t] = sum(v for batch in (first, second) for j, v in batch.items() if j in beans)
        for start in range(2023, 2029):
            if sum(bean_area[t] for t in range(start, start + 3)) + EPS < size:
                raise ValueError(f"三年豆类覆盖面积不足：{start}, {p}")
    yearly = {t: {"收入": 0.0, "成本": cost[t], "原价量": 0.0, "超产量": 0.0} for t in YEARS}
    detail = []
    for (t, s, j), q in sorted(market.items()):
        normal = min(q, demand[s, j])
        extra = q - normal
        revenue = (normal + discount * extra) * price[s, j]
        yearly[t]["原价量"] += normal
        yearly[t]["超产量"] += extra
        yearly[t]["收入"] += revenue
        detail.append((t, s, j, crops[j], q, demand[s, j], normal, extra, price[s, j], revenue))
    for row in yearly.values():
        row["利润"] = row["收入"] - row["成本"]
    return yearly, detail


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--seconds", type=float, default=120, help="每个情景的求解时间上限")
    parser.add_argument("--gap", type=float, default=0.01, help="相对最优性间隙目标")
    parser.add_argument("--min-fraction", type=float, default=0.1, help="单作物最小占地比例")
    parser.add_argument("--max-plots", type=int, default=12, help="同作物同季最多地块数")
    parser.add_argument("--scenario", choices=("both", "1", "2"), default="both")
    args = parser.parse_args()
    if not (0 < args.min_fraction <= 1 and args.max_plots > 0 and args.seconds > 0
            and 0 <= args.gap < 1):
        parser.error("检查时间、间隙、面积比例和地块数取值")
    data = read_data(args.data)
    plots, crops, _, stats, _, history, demand = data
    args.out.mkdir(parents=True, exist_ok=True)
    for i, discount in enumerate((0, 0.5), 1):
        if args.scenario not in ("both", str(i)):
            continue
        stem = f"result1_{i}"
        m, x, sales = build(data, discount, args.min_fraction, args.max_plots)
        result, violation, integrality = m.solve(args.seconds, args.gap)
        areas = {key: float(result.x[index]) for key, index in x.items() if result.x[index] > EPS}
        yearly, detail = verify(data, areas, discount, args.min_fraction, args.max_plots)
        profit = sum(row["利润"] for row in yearly.values())
        if not math.isclose(profit, -result.fun, rel_tol=1e-9, abs_tol=0.1):
            raise RuntimeError(f"独立重算利润与求解器目标不一致：{profit}, {-result.fun}")
        market_rows = {(t, s, j): normal for t, s, j, _, _, _, normal, *_ in detail}
        if any(abs(result.x[idx] - market_rows.get(key, 0)) > 0.05 for key, idx in sales.items()):
            raise RuntimeError("原价销量核对失败")
        write_csv(args.out / f"{stem}_area.csv", ("年份", "地块", "季槽", "作物编号", "作物", "面积/亩"),
                  [(t, p, s, j, crops[j], a) for (t, p, s, j), a in sorted(areas.items())])
        write_csv(args.out / f"{stem}_sales.csv",
                  ("年份", "季槽", "作物编号", "作物", "产量/斤", "销量上限/斤", "原价量/斤",
                   "超产量/斤", "原价/元每斤", "收入/元"), detail)
        report = {"情景": i, "超产价格比例": discount, "2023种植记录数": len(history),
                  "地块数": len(plots), "作物数": len(crops), "适种参数数": len(stats),
                  "变量数": len(m.lower), "二元变量数": sum(m.integer), "约束数": len(m.lo),
                  "时间上限/秒": args.seconds, "目标间隙": args.gap,
                  "最小面积比例": args.min_fraction, "每作物每季地块数上限": args.max_plots,
                  "状态码": int(result.status), "求解器状态": result.message,
                  "利润/元": profit, "利润上界/元": float(-result.mip_dual_bound),
                  "实际相对间隙": float(result.mip_gap), "最大约束误差": violation,
                  "最大整数误差": integrality, "逐年结果": yearly}
        (args.out / f"{stem}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(stem, f"利润 {profit:,.2f} 元，间隙 {result.mip_gap:.2%}，{result.message}", flush=True)


if __name__ == "__main__":
    run()
