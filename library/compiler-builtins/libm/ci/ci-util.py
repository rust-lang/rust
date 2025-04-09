#!/usr/bin/env python3
"""Utilities for CI.

This dynamically prepares a list of routines that had a source file change based on
git history.
"""

import json
import os
import subprocess as sp
import sys
from dataclasses import dataclass
from glob import glob, iglob
from inspect import cleandoc
from os import getenv
from pathlib import Path
from typing import TypedDict, Self

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

        check-regressions [--home iai-home] [--allow-pr-override pr_number]
            Check `iai-home` (or `iai-home` if unspecified) for `summary.json`
            files and see if there are any regressions. This is used as a workaround
            for `iai-callgrind` not exiting with error status; see
            <https://github.com/iai-callgrind/iai-callgrind/issues/337>.

            If `--allow-pr-override` is specified, the regression check will not exit
            with failure if any line in the PR starts with `allow-regressions`.
    """
)

REPO_ROOT = Path(__file__).parent.parent
GIT = ["git", "-C", REPO_ROOT]
DEFAULT_BRANCH = "master"
WORKFLOW_NAME = "CI"  # Workflow that generates the benchmark artifacts
ARTIFACT_GLOB = "baseline-icount*"
# Place this in a PR body to skip regression checks (must be at the start of a line).
REGRESSION_DIRECTIVE = "ci: allow-regressions"
# Place this in a PR body to skip extensive tests
SKIP_EXTENSIVE_DIRECTIVE = "ci: skip-extensive"
# Place this in a PR body to allow running a large number of extensive tests. If not
# set, this script will error out if a threshold is exceeded in order to avoid
# accidentally spending huge amounts of CI time.
ALLOW_MANY_EXTENSIVE_DIRECTIVE = "ci: allow-many-extensive"
MANY_EXTENSIVE_THRESHOLD = 20

# Don't run exhaustive tests if these files change, even if they contaiin a function
# definition.
IGNORE_FILES = [
    "libm/src/math/support/",
    "libm/src/libm_helper.rs",
    "libm/src/math/arch/intrinsics.rs",
]

TYPES = ["f16", "f32", "f64", "f128"]


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


@dataclass
class PrInfo:
    """GitHub response for PR query"""

    body: str
    commits: list[str]
    created_at: str
    number: int

    @classmethod
    def load(cls, pr_number: int | str) -> Self:
        """For a given PR number, query the body and commit list"""
        pr_info = sp.check_output(
            [
                "gh",
                "pr",
                "view",
                str(pr_number),
                "--json=number,commits,body,createdAt",
                # Flatten the commit list to only hashes, change a key to snake naming
                "--jq=.commits |= map(.oid) | .created_at = .createdAt | del(.createdAt)",
            ],
            text=True,
        )
        eprint("PR info:", json.dumps(pr_info, indent=4))
        return cls(**json.loads(pr_info))

    def contains_directive(self, directive: str) -> bool:
        """Return true if the provided directive is on a line in the PR body"""
        lines = self.body.splitlines()
        return any(line.startswith(directive) for line in lines)


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

        ret: dict[str, list[str]] = {}
        for r in sorted(routines):
            ret.setdefault(self.defs[r]["type"], []).append(r)

        return ret

    def make_workflow_output(self) -> str:
        """Create a JSON object a list items for each type's changed files, if any
        did change, and the routines that were affected by the change.
        """

        pr_number = os.environ.get("PR_NUMBER")
        skip_tests = False
        error_on_many_tests = False

        if pr_number is not None and len(pr_number) > 0:
            pr = PrInfo.load(pr_number)
            skip_tests = pr.contains_directive(SKIP_EXTENSIVE_DIRECTIVE)
            error_on_many_tests = not pr.contains_directive(
                ALLOW_MANY_EXTENSIVE_DIRECTIVE
            )

            if skip_tests:
                eprint("Skipping all extensive tests")

        changed = self.changed_routines()
        ret = []
        total_to_test = 0

        for ty in TYPES:
            ty_changed = changed.get(ty, [])
            ty_to_test = [] if skip_tests else ty_changed
            total_to_test += len(ty_to_test)

            item = {
                "ty": ty,
                "changed": ",".join(ty_changed),
                "to_test": ",".join(ty_to_test),
            }

            ret.append(item)
        output = json.dumps({"matrix": ret}, separators=(",", ":"))
        eprint(f"output: {output}")
        eprint(f"total extensive tests: {total_to_test}")

        if error_on_many_tests and total_to_test > MANY_EXTENSIVE_THRESHOLD:
            eprint(
                f"More than {MANY_EXTENSIVE_THRESHOLD} tests would be run; add"
                f" `{ALLOW_MANY_EXTENSIVE_DIRECTIVE}` to the PR body if this is intentional"
            )
            exit(1)

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
                "--status=success",
                f"--branch={DEFAULT_BRANCH}",
                "--json=databaseId,url,headSha,conclusion,createdAt,"
                "status,workflowDatabaseId,workflowName",
                # Return the first array element matching our workflow name. NB: cannot
                # just use `--limit=1`, jq filtering happens after limiting. We also
                # cannot just use `--workflow` because GH gets confused from
                # different file names in history.
                f'--jq=[.[] | select(.workflowName == "{WORKFLOW_NAME}")][0]',
            ],
            text=True,
        )
    except sp.CalledProcessError as e:
        eprint(f"failed to run github command: {e}")
        return

    try:
        latest = json.loads(latest_job)
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


def check_iai_regressions(args: list[str]):
    """Find regressions in iai summary.json files, exit with failure if any are
    found.
    """

    iai_home_str = "iai-home"
    pr_number = None

    while len(args) > 0:
        match args:
            case ["--home", home, *rest]:
                iai_home_str = home
                args = rest
            case ["--allow-pr-override", pr_num, *rest]:
                pr_number = pr_num
                args = rest
            case _:
                eprint(USAGE)
                exit(1)

    iai_home = Path(iai_home_str)

    found_summaries = False
    regressions: list[dict] = []
    for summary_path in iglob("**/summary.json", root_dir=iai_home, recursive=True):
        found_summaries = True
        with open(iai_home / summary_path, "r") as f:
            summary = json.load(f)

        summary_regs = []
        run = summary["callgrind_summary"]["callgrind_run"]
        fname = summary["function_name"]
        id = summary["id"]
        name_entry = {"name": f"{fname}.{id}"}

        for segment in run["segments"]:
            summary_regs.extend(segment["regressions"])

        summary_regs.extend(run["total"]["regressions"])

        regressions.extend(name_entry | reg for reg in summary_regs)

    if not found_summaries:
        eprint(f"did not find any summary.json files within {iai_home}")
        exit(1)

    if len(regressions) == 0:
        eprint("No regressions found")
        return

    eprint("Found regressions:", json.dumps(regressions, indent=4))

    if pr_number is not None:
        pr = PrInfo.load(pr_number)
        if pr.contains_directive(REGRESSION_DIRECTIVE):
            eprint("PR allows regressions, returning")
            return

    exit(1)


def main():
    match sys.argv[1:]:
        case ["generate-matrix"]:
            ctx = Context()
            output = ctx.make_workflow_output()
            print(f"matrix={output}")
        case ["locate-baseline", *flags]:
            locate_baseline(flags)
        case ["check-regressions", *args]:
            check_iai_regressions(args)
        case ["--help" | "-h"]:
            print(USAGE)
            exit()
        case _:
            eprint(USAGE)
            exit(1)


if __name__ == "__main__":
    main()
