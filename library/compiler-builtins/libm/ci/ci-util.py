#!/usr/bin/env python3
"""Utilities for CI.

This dynamically prepares a list of routines that had a source file change based on
git history.
"""

import json
import subprocess as sp
import sys
from dataclasses import dataclass
from glob import glob, iglob
from inspect import cleandoc
from os import getenv
from pathlib import Path
from typing import TypedDict

USAGE = cleandoc(
    """
    usage:

    ./ci/ci-util.py <COMMAND> [flags]

    COMMAND:
        generate-matrix
            Calculate a matrix of which functions had source change, print that as
             a JSON object.

        locate-baseline [--download] [--extract]
            Locate the most recent benchmark baseline available in CI and, if flags
            specify, download and extract it. Never exits with nonzero status if
            downloading fails.

            Note that `--extract` will overwrite files in `iai-home`.

        check-regressions [iai-home]
            Check `iai-home` (or `iai-home` if unspecified) for `summary.json`
            files and see if there are any regressions. This is used as a workaround
            for `iai-callgrind` not exiting with error status; see
            <https://github.com/iai-callgrind/iai-callgrind/issues/337>.
    """
)

REPO_ROOT = Path(__file__).parent.parent
GIT = ["git", "-C", REPO_ROOT]
DEFAULT_BRANCH = "master"
WORKFLOW_NAME = "CI"  # Workflow that generates the benchmark artifacts
ARTIFACT_GLOB = "baseline-icount*"

# Don't run exhaustive tests if these files change, even if they contaiin a function
# definition.
IGNORE_FILES = [
    "src/math/support/",
    "src/libm_helper.rs",
    "src/math/arch/intrinsics.rs",
]

TYPES = ["f16", "f32", "f64", "f128"]


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


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


def locate_baseline(flags: list[str]) -> None:
    """Find the most recent baseline from CI, download it if specified.

    This returns rather than erroring, even if the `gh` commands fail. This is to avoid
    erroring in CI if the baseline is unavailable (artifact time limit exceeded, first
    run on the branch, etc).
    """

    download = False
    extract = False

    while len(flags) > 0:
        match flags[0]:
            case "--download":
                download = True
            case "--extract":
                extract = True
            case _:
                eprint(USAGE)
                exit(1)
        flags = flags[1:]

    if extract and not download:
        eprint("cannot extract without downloading")
        exit(1)

    try:
        # Locate the most recent job to complete with success on our branch
        latest_job = sp.check_output(
            [
                "gh",
                "run",
                "list",
                "--limit=1",
                "--status=success",
                f"--branch={DEFAULT_BRANCH}",
                "--json=databaseId,url,headSha,conclusion,createdAt,"
                "status,workflowDatabaseId,workflowName",
                f'--jq=select(.[].workflowName == "{WORKFLOW_NAME}")',
            ],
            text=True,
        )
        eprint(f"latest: '{latest_job}'")
    except sp.CalledProcessError as e:
        eprint(f"failed to run github command: {e}")
        return

    try:
        latest = json.loads(latest_job)[0]
        eprint("latest job: ", json.dumps(latest, indent=4))
    except json.JSONDecodeError as e:
        eprint(f"failed to decode json '{latest_job}', {e}")
        return

    if not download:
        eprint("--download not specified, returning")
        return

    job_id = latest.get("databaseId")
    if job_id is None:
        eprint("skipping download step")
        return

    sp.run(
        ["gh", "run", "download", str(job_id), f"--pattern={ARTIFACT_GLOB}"],
        check=False,
    )

    if not extract:
        eprint("skipping extraction step")
        return

    # Find the baseline with the most recent timestamp. GH downloads the files to e.g.
    # `some-dirname/some-dirname.tar.xz`, so just glob the whole thing together.
    candidate_baselines = glob(f"{ARTIFACT_GLOB}/{ARTIFACT_GLOB}")
    if len(candidate_baselines) == 0:
        eprint("no possible baseline directories found")
        return

    candidate_baselines.sort(reverse=True)
    baseline_archive = candidate_baselines[0]
    eprint(f"extracting {baseline_archive}")
    sp.run(["tar", "xJvf", baseline_archive], check=True)
    eprint("baseline extracted successfully")


def check_iai_regressions(iai_home: str | None | Path):
    """Find regressions in iai summary.json files, exit with failure if any are
    found.
    """
    if iai_home is None:
        iai_home = "iai-home"
    iai_home = Path(iai_home)

    found_summaries = False
    regressions = []
    for summary_path in iglob("**/summary.json", root_dir=iai_home, recursive=True):
        found_summaries = True
        with open(iai_home / summary_path, "r") as f:
            summary = json.load(f)

        summary_regs = []
        run = summary["callgrind_summary"]["callgrind_run"]
        name_entry = {"name": f"{summary["function_name"]}.{summary["id"]}"}

        for segment in run["segments"]:
            summary_regs.extend(segment["regressions"])

        summary_regs.extend(run["total"]["regressions"])

        regressions.extend(name_entry | reg for reg in summary_regs)

    if not found_summaries:
        eprint(f"did not find any summary.json files within {iai_home}")
        exit(1)

    if len(regressions) > 0:
        eprint("Found regressions:", json.dumps(regressions, indent=4))
        exit(1)


def main():
    match sys.argv[1:]:
        case ["generate-matrix"]:
            ctx = Context()
            output = ctx.make_workflow_output()
            print(f"matrix={output}")
        case ["locate-baseline", *flags]:
            locate_baseline(flags)
        case ["check-regressions"]:
            check_iai_regressions(None)
        case ["check-regressions", iai_home]:
            check_iai_regressions(iai_home)
        case ["--help" | "-h"]:
            print(USAGE)
            exit()
        case _:
            eprint(USAGE)
            exit(1)


if __name__ == "__main__":
    main()
