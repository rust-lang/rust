#!/usr/bin/env python3
"""Utilities for CI.

This dynamically prepares a list of routines that had a source file change based on
git history.
"""

import json
import os
import re
import subprocess as sp
import sys
from dataclasses import dataclass
from glob import glob
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

        locate-baseline [--download] [--extract] [--tag TAG]
            Locate the most recent benchmark baseline available in CI and, if flags
            specify, download and extract it. Never exits with nonzero status if
            downloading fails.

            `--tag` can be specified to look for artifacts with a specific tag, such as
            for a specific architecture.

            Note that `--extract` will overwrite files in `iai-home`.

        handle-bench-regressions PR_NUMBER
            Exit with success if the pull request contains a line starting with
            `ci: allow-regressions`, indicating that regressions in benchmarks should
            be accepted. Otherwise, exit 1.
    """
)

REPO_ROOT = Path(__file__).parent.parent
GIT = ["git", "-C", REPO_ROOT]
DEFAULT_BRANCH = "master"
WORKFLOW_NAME = "CI"  # Workflow that generates the benchmark artifacts
ARTIFACT_PREFIX = "baseline-icount*"
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

# libm PR CI takes a long time and doesn't need to run unless relevant files have been
# changed. Anything matching this regex pattern will trigger a run.
TRIGGER_LIBM_PR_CI = ".*(libm|musl).*"

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
        if not self.is_pr():
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

    def is_pr(self) -> bool:
        """Check if we are looking at a PR rather than a push."""
        return self.gh_ref is not None and "merge" in self.gh_ref

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

    def may_skip_libm_ci(self) -> bool:
        """If this is a PR and no libm files were changed, allow skipping libm
        jobs."""

        if self.is_pr():
            return all(not re.match(TRIGGER_LIBM_PR_CI, str(f)) for f in self.changed)

        return False

    def emit_workflow_output(self):
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
        matrix = []
        total_to_test = 0

        # Figure out which extensive tests need to run
        for ty in TYPES:
            ty_changed = changed.get(ty, [])
            ty_to_test = [] if skip_tests else ty_changed
            total_to_test += len(ty_to_test)

            item = {
                "ty": ty,
                "changed": ",".join(ty_changed),
                "to_test": ",".join(ty_to_test),
            }

            matrix.append(item)

        ext_matrix = json.dumps({"extensive_matrix": matrix}, separators=(",", ":"))
        may_skip = str(self.may_skip_libm_ci()).lower()
        print(f"extensive_matrix={ext_matrix}")
        print(f"may_skip_libm_ci={may_skip}")
        eprint(f"extensive_matrix={ext_matrix}")
        eprint(f"may_skip_libm_ci={may_skip}")
        eprint(f"total extensive tests: {total_to_test}")

        if error_on_many_tests and total_to_test > MANY_EXTENSIVE_THRESHOLD:
            eprint(
                f"More than {MANY_EXTENSIVE_THRESHOLD} tests would be run; add"
                f" `{ALLOW_MANY_EXTENSIVE_DIRECTIVE}` to the PR body if this is"
                " intentional. If this is refactoring that happens to touch a lot of"
                f" files, `{SKIP_EXTENSIVE_DIRECTIVE}` can be used instead."
            )
            exit(1)


def locate_baseline(flags: list[str]) -> None:
    """Find the most recent baseline from CI, download it if specified.

    This returns rather than erroring, even if the `gh` commands fail. This is to avoid
    erroring in CI if the baseline is unavailable (artifact time limit exceeded, first
    run on the branch, etc).
    """

    download = False
    extract = False
    tag = ""

    while len(flags) > 0:
        match flags[0]:
            case "--download":
                download = True
            case "--extract":
                extract = True
            case "--tag":
                tag = flags[1]
                flags = flags[1:]
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

    artifact_glob = f"{ARTIFACT_PREFIX}{f"-{tag}" if tag else ""}*"

    sp.run(
        ["gh", "run", "download", str(job_id), f"--pattern={artifact_glob}"],
        check=False,
    )

    if not extract:
        eprint("skipping extraction step")
        return

    # Find the baseline with the most recent timestamp. GH downloads the files to e.g.
    # `some-dirname/some-dirname.tar.xz`, so just glob the whole thing together.
    candidate_baselines = glob(f"{artifact_glob}/{artifact_glob}")
    if len(candidate_baselines) == 0:
        eprint("no possible baseline directories found")
        return

    candidate_baselines.sort(reverse=True)
    baseline_archive = candidate_baselines[0]
    eprint(f"extracting {baseline_archive}")
    sp.run(["tar", "xJvf", baseline_archive], check=True)
    eprint("baseline extracted successfully")


def handle_bench_regressions(args: list[str]):
    """Exit with error unless the PR message contains an ignore directive."""

    match args:
        case [pr_number]:
            pr_number = pr_number
        case _:
            eprint(USAGE)
            exit(1)

    pr = PrInfo.load(pr_number)
    if pr.contains_directive(REGRESSION_DIRECTIVE):
        eprint("PR allows regressions")
        return

    eprint("Regressions were found; benchmark failed")
    exit(1)


def main():
    match sys.argv[1:]:
        case ["generate-matrix"]:
            ctx = Context()
            ctx.emit_workflow_output()
        case ["locate-baseline", *flags]:
            locate_baseline(flags)
        case ["handle-bench-regressions", *args]:
            handle_bench_regressions(args)
        case ["--help" | "-h"]:
            print(USAGE)
            exit()
        case _:
            eprint(USAGE)
            exit(1)


if __name__ == "__main__":
    main()
