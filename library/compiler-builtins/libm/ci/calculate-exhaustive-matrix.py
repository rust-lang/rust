#!/usr/bin/env python3
"""Calculate which exhaustive tests should be run as part of CI.

This dynamically prepares a list of routines that had a source file change based on
git history.
"""

import subprocess as sp
import sys
import json
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import TypedDict


REPO_ROOT = Path(__file__).parent.parent
GIT = ["git", "-C", REPO_ROOT]

# Don't run exhaustive tests if these files change, even if they contaiin a function
# definition.
IGNORE_FILES = [
    "src/math/support/",
    "src/libm_helper.rs",
    "src/math/arch/intrinsics.rs",
]

TYPES = ["f16", "f32", "f64", "f128"]


class FunctionDef(TypedDict):
    """Type for an entry in `function-definitions.json`"""

    sources: list[str]
    type: str


@dataclass
class Context:
    gh_ref: str | None
    changed: list[Path]
    defs: dict[str, FunctionDef]

    def __init__(self) -> None:
        self.gh_ref = getenv("GITHUB_REF")
        self.changed = []
        self._init_change_list()

        with open(REPO_ROOT.joinpath("etc/function-definitions.json")) as f:
            defs = json.load(f)

        defs.pop("__comment", None)
        self.defs = defs

    def _init_change_list(self):
        """Create a list of files that have been changed. This uses GITHUB_REF if
        available, otherwise a diff between `HEAD` and `master`.
        """

        # For pull requests, GitHub creates a ref `refs/pull/1234/merge` (1234 being
        # the PR number), and sets this as `GITHUB_REF`.
        ref = self.gh_ref
        eprint(f"using ref `{ref}`")
        if ref is None or "merge" not in ref:
            # If the ref is not for `merge` then we are not in PR CI
            eprint("No diff available for ref")
            return

        # The ref is for a dummy merge commit. We can extract the merge base by
        # inspecting all parents (`^@`).
        merge_sha = sp.check_output(
            GIT + ["show-ref", "--hash", ref], text=True
        ).strip()
        merge_log = sp.check_output(GIT + ["log", "-1", merge_sha], text=True)
        eprint(f"Merge:\n{merge_log}\n")

        parents = (
            sp.check_output(GIT + ["rev-parse", f"{merge_sha}^@"], text=True)
            .strip()
            .splitlines()
        )
        assert len(parents) == 2, f"expected two-parent merge but got:\n{parents}"
        base = parents[0].strip()
        incoming = parents[1].strip()

        eprint(f"base: {base}, incoming: {incoming}")
        textlist = sp.check_output(
            GIT + ["diff", base, incoming, "--name-only"], text=True
        )
        self.changed = [Path(p) for p in textlist.splitlines()]

    @staticmethod
    def _ignore_file(fname: str) -> bool:
        return any(fname.startswith(pfx) for pfx in IGNORE_FILES)

    def changed_routines(self) -> dict[str, list[str]]:
        """Create a list of routines for which one or more files have been updated,
        separated by type.
        """
        routines = set()
        for name, meta in self.defs.items():
            # Don't update if changes to the file should be ignored
            sources = (f for f in meta["sources"] if not self._ignore_file(f))

            # Select changed files
            changed = [f for f in sources if Path(f) in self.changed]

            if len(changed) > 0:
                eprint(f"changed files for {name}: {changed}")
                routines.add(name)

        ret = {}
        for r in sorted(routines):
            ret.setdefault(self.defs[r]["type"], []).append(r)

        return ret

    def make_workflow_output(self) -> str:
        """Create a JSON object a list items for each type's changed files, if any
        did change, and the routines that were affected by the change.
        """
        changed = self.changed_routines()
        ret = []
        for ty in TYPES:
            ty_changed = changed.get(ty, [])
            item = {
                "ty": ty,
                "changed": ",".join(ty_changed),
            }
            ret.append(item)
        output = json.dumps({"matrix": ret}, separators=(",", ":"))
        eprint(f"output: {output}")
        return output


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def main():
    ctx = Context()
    output = ctx.make_workflow_output()
    print(f"matrix={output}")


if __name__ == "__main__":
    main()
