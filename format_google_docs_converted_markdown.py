import re
import os
import sys

LINE_REGEX_COLORS = {
    "sup_number": {
        "search_regex": r"( )(\d+)(。)",
        "replace_regex": r'\1<sup><font color="">\2<color></font></sup>\3',
        "color": "red",
    },
    "h2_title": {
        "search_regex": r"^(##\s+\*\*)(.*)(\*\*)",
        "replace_regex": r'\1<font color="">\2</font>\3',
        "color": "DodgerBlue",
    },
    "h3_title": {
        "search_regex": r"^(###\s+\*\*)(.*)(\*\*)",
        "replace_regex": r'\1<font color="">\2</font>\3',
        "color": "DarkViolet",
    },
    "h4_reference_title": {
        "search_regex": r"^(####\s+\*\*)(引用的著作)(\*\*)",
        "replace_regex": r'\1<font color="">引用的资料</font>\3',
        "color": "LightSeaGreen",
    }
}


def get_rules(key: str) -> tuple[str, str]:
    color = LINE_REGEX_COLORS[key]["color"]
    patten = LINE_REGEX_COLORS[key]["search_regex"]
    repl = LINE_REGEX_COLORS[key]["replace_regex"].replace(
        'color=""', f'color="{color}"'
    )
    return (patten, repl)


def format_lines(lines: list[str], key: str):
    patten, repl = get_rules(key)
    for index in range(0, len(lines)):
        if re.search(pattern=patten, string=lines[index]):
            lines[index] = re.sub(pattern=patten, repl=repl, string=lines[index])


def format_sup_numbers(lines: list[str]):
    format_lines(lines=lines, key="sup_number")


def format_h2_titles(lines: list[str]):
    format_lines(lines=lines, key="h2_title")


def format_h3_titles(lines: list[str]):
    format_lines(lines=lines, key="h3_title")


def format_h4_reference_title(lines: list[str]):
    format_lines(lines=lines, key="h4_reference_title")


def format_all(path: str):
    with open(path, encoding="utf-8") as fr:
        lines = fr.readlines()
        format_sup_numbers(lines=lines)
        format_h2_titles(lines=lines)
        format_h3_titles(lines=lines)
        format_h4_reference_title(lines=lines)

    n, e = os.path.splitext(path)
    with open(f"{n}.repl{e}", mode="w", encoding="utf-8") as fw:
        fw.writelines(lines)


if __name__ == "__main__":
    format_all(path=sys.argv[1])
