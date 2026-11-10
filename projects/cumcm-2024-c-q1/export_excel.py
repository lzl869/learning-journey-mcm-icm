"""将求解后的面积 CSV 写入模板，保留模板样式、表头、合并区域与页脚。"""
import argparse
import csv
import math
import posixpath
import re
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from solve import DRY, ROOT, YEARS, read_data

# 仅替换模板已有的空白单元格，保留其属性和其余 XML。
BLANK_CELL = re.compile(rb'<c\s+r="([A-Z]+[0-9]+)"([^<>]*?)/>')
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def worksheet_paths(archive):
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {row.get("Id"): row.get("Target") for row in rels}
    paths = {}
    for sheet in workbook.findall(".//{*}sheet"):
        target = targets[sheet.get(f"{{{REL}}}id")]
        paths[sheet.get("name")] = (target.lstrip("/") if target.startswith("/")
                                    else posixpath.normpath(posixpath.join("xl", target)))
    return paths


def workbook_layout(path, plots, crops, blank=False):
    # 仅用于读取坐标和复核数值。页脚原样留在 ZIP 中。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot parse header or footer")
        book = load_workbook(path, data_only=False)
    if book.sheetnames != [str(t) for t in YEARS]:
        raise ValueError(f"模板年份页不符：{path}")
    lookup = {}
    for sheet in book:
        year = int(sheet.title)
        columns = {str(cell.value).strip(): cell.column for cell in sheet[1][2:]
                   if cell.value is not None}
        if set(columns) != set(crops.values()):
            raise ValueError(f"作物列不完整：{sheet.title}")
        season = 0
        seen = set()
        for row in range(2, sheet.max_row + 1):
            tag = "".join(str(sheet.cell(row, 1).value or "").split())
            if tag in ("第一季", "第二季"):
                season = 1 if tag == "第一季" else 2
            plot = sheet.cell(row, 2).value
            if plot not in plots:
                continue
            if not season or (plot, season) in seen:
                raise ValueError(f"季次或地块行重复：{year}, {plot}, {season}")
            seen.add((plot, season))
            for crop, name in crops.items():
                col = columns[name]
                cell = sheet.cell(row, col)
                if blank and cell.value is not None:
                    raise ValueError(f"模板含已有结果：{year}!{cell.coordinate}")
                lookup[year, plot, season, crop] = (f"{get_column_letter(col)}{row}", cell.value)
        expected = {(plot, s) for plot, (kind, _) in plots.items()
                    for s in ((1,) if kind in DRY else (1, 2))}
        if seen != expected:
            raise ValueError(f"模板地块行不完整：{year}")
    book.close()
    return lookup


def export(template, target, csv_path, plots, crops):
    if template.resolve() == target.resolve():
        raise ValueError("输出位置必须与原始模板分开")
    layout = workbook_layout(template, plots, crops, blank=True)
    filled = defaultdict(dict)
    areas = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["年份"]), row["地块"], int(row["季槽"]), int(row["作物编号"]))
            area = float(row["面积/亩"])
            if key not in layout or key in areas or not math.isfinite(area) or area <= 0:
                raise ValueError(f"面积明细无效：{key}, {area}")
            areas[key] = area
            filled[str(key[0])][layout[key][0].encode()] = area

    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        temp = Path(tmp) / target.name
        with ZipFile(template) as source, ZipFile(temp, "w") as output:
            patches = {path: filled[year].copy() for year, path in worksheet_paths(source).items()}
            for member in source.infolist():
                content = source.read(member.filename)
                if member.filename in patches:
                    patch = patches[member.filename]

                    def replace(match):
                        coordinate, attrs = match.groups()
                        if coordinate not in patch:
                            return match.group()
                        area = format(patch.pop(coordinate), ".16g").encode()
                        return b'<c r="' + coordinate + b'"' + attrs + b'><v>' + area + b'</v></c>'

                    content = BLANK_CELL.sub(replace, content)
                    if patch:
                        raise ValueError(f"模板缺少可写空单元格：{list(patch)[:5]}")
                output.writestr(member, content)
        # 回读全部七年，逐格核对正面积，同时确认未种植单元格仍为空。
        written = workbook_layout(temp, plots, crops)
        for key, (_, value) in written.items():
            if key in areas:
                if not isinstance(value, (float, int)) or not math.isclose(
                    value, areas[key], abs_tol=1e-9, rel_tol=0
                ):
                    raise ValueError(f"Excel 回读数值不一致：{key}")
            elif value is not None:
                raise ValueError(f"Excel 多写单元格：{key}")
        temp.replace(target)
    print(f"{target.name} 已写入并核对 {len(areas)} 个正面积单元格")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--templates", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    plots, crops, *_ = read_data(args.data)
    templates = args.templates or args.data / "templates"
    for i in (1, 2):
        stem = f"result1_{i}"
        export(templates / f"{stem}.xlsx", args.out / f"{stem}.xlsx",
               args.out / f"{stem}_area.csv", plots, crops)


if __name__ == "__main__":
    main()
