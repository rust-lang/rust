#!/usr/bin/env python3
"""Create beads issues from a markdown list of Triton ops.

Expected markdown format includes bullets with backticked ops, e.g.:
  - `tt.load`
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TT_OP_RE = re.compile(r"`(tt\.[a-zA-Z0-9_.]+)`")


def extract_tt_ops(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    ops = sorted(set(TT_OP_RE.findall(text)))
    return ops


def create_issue(
    op: str,
    title_template: str,
    description_template: str,
    issue_type: str,
    priority: str,
    labels: str,
    dry_run: bool,
) -> None:
    title = title_template.format(op=op)
    description = description_template.format(op=op)

    cmd = [
        "bd",
        "create",
        "--title",
        title,
        "--description",
        description,
        "--type",
        issue_type,
        "--priority",
        priority,
        "--labels",
        labels,
    ]
    if dry_run:
        cmd.append("--dry-run")

    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create one beads issue per `tt.*` op found in a markdown file.")
    )
    parser.add_argument(
        "--md-file",
        default=".ai/specs/triton/triton.dialect.ops.md",
        help="Path to markdown file containing Triton ops.",
    )
    parser.add_argument(
        "--title-template",
        default="Implement support for {op}",
        help="Issue title template. Supports placeholder: {op}",
    )
    parser.add_argument(
        "--description-template",
        default=(
            "Track work for `{op}`.\n\n"
            "## Context\n"
            "- Placeholder context for {op}\n\n"
            "## TODO\n"
            "- Replace this template content with final details.\n"
        ),
        help="Issue description template. Supports placeholder: {op}",
    )
    parser.add_argument(
        "--type",
        default="task",
        help="Issue type for bd create (default: task).",
    )
    parser.add_argument(
        "--priority",
        default="2",
        help="Issue priority for bd create (default: 2).",
    )
    parser.add_argument(
        "--labels",
        default="needs-planning",
        help="Comma-separated labels for bd create (default: needs-planning).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create issues. If omitted, runs in --dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    md_file = Path(args.md_file)
    if not md_file.exists():
        print(f"error: markdown file not found: {md_file}", file=sys.stderr)
        return 1

    ops = extract_tt_ops(md_file)
    if not ops:
        print(f"error: no tt.* ops found in: {md_file}", file=sys.stderr)
        return 1

    dry_run = not args.apply
    mode = "DRY RUN" if dry_run else "APPLY"
    print(f"[{mode}] Found {len(ops)} tt ops in {md_file}")

    for op in ops:
        print(f"- creating issue for {op}")
        create_issue(
            op=op,
            title_template=args.title_template,
            description_template=args.description_template,
            issue_type=args.type,
            priority=args.priority,
            labels=args.labels,
            dry_run=dry_run,
        )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
