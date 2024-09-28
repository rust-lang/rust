#!/usr/bin/env python3
"""
Tool for updating `Cargo.lock`s in an automated way.

This is used by the automated weekly update CI job


Update multiple lockfiles in the repository and record the changes to
`update_output.json`.
"""

import json
import os
import subprocess as sp
import sys
import tomllib
from pathlib import Path
from inspect import cleandoc

LOCKFILES = {
    "root": ".",
    "library": "library",
    "rustbook": "src/tools/rustbook",
    "bootstrap": "src/bootstrap",
}

UPDATE_JSON_OUTFILE = "update_output.json"
COMMIT_TEXT_OUTFILE = "commit.txt"
PR_BODY_OUTFILE = "pr_body.md"

USAGE = cleandoc(
    f"""
    Update all lockfiles in the repository

    Usage: ./src/ci/scripts/update-all-lockfiles.py <OPTION>

    Options:
        --run-update                Update all lockfiles and save output to
                                    {UPDATE_JSON_OUTFILE}
        --print-names               Print a list of identifiers for lockfiles
        --list-skip-branches        Print a list of branches that are in the
                                    process of being merged, so do not need to
                                    be updated
        --prepare-pr-files <NAME>   Prepare {COMMIT_TEXT_OUTFILE} and {PR_BODY_OUTFILE}
                                    for branch NAME. Needs {UPDATE_JSON_OUTFILE}.
        --restore-lockfile <NAME>   Restore the lockfile for NAME, using the contents
                                    and path in {UPDATE_JSON_OUTFILE}
    """
)

COMMIT_TEMPLATE = cleandoc(
    """
    {name}: run `cargo update`

    Dependency count before update: {old_dep_count}
    Dependency count after update: {new_dep_count}

    Log:

    ```text
    {log}
    ```
    """
)

PR_BODY_TEMPLATE = cleandoc(
    """
    Automation to keep dependencies in `Cargo.lock` current.

    Dependency count before update: {old_dep_count}
    Dependency count after update: {new_dep_count}

    The following is the output from `cargo update` in `{path}`:

    ```text
    {log}
    ```
    """
)


def tee(args: list[str | Path]) -> str:
    """Forward stdout and stderr to the user but also append it to `s`"""
    s = ""
    proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    for line in proc.stdout.readlines():
        sys.stdout.write(line)

        # Remove first line that just says it is updating the index
        if "crates.io index" not in line:
            s += line

    return s.rstrip()


def branch_name(name: str) -> str:
    """Get the name of a branch from a lockfile's name"""
    return f"cargo-update-{name}"


def count_lockfile_dependencies(contents: str) -> int:
    """Given a lockfile's contents, figure out how many dependencies it contains"""
    data = tomllib.loads(contents)
    return len(data["package"])


def run_update():
    """Update all lockfiles in the repository"""
    output = {}

    for name, path in LOCKFILES.items():
        path = Path(path)
        manifest_path = path / "Cargo.toml"
        lockfile_path = path / "Cargo.lock"

        old_dep_count = count_lockfile_dependencies(lockfile_path.read_text())

        log = f"{name} dependencies:"
        print(log)

        log += "\n"
        log += tee(["cargo", "update", "--manifest-path", manifest_path])

        print()

        new_lockfile = lockfile_path.read_text()
        new_dep_count = count_lockfile_dependencies(new_lockfile)

        single_item = {
            "branch": branch_name(name),
            "pr_title": f"Weekly {name} `cargo update`",
            "path": f"{path}",
            "old_dep_count": old_dep_count,
            "new_dep_count": new_dep_count,
            "lockfile_path": f"{lockfile_path}",
            "lockfile": new_lockfile,
            "log": log,
        }

        output[name] = single_item

    with open(UPDATE_JSON_OUTFILE, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Wrote output to {UPDATE_JSON_OUTFILE}")


def print_names():
    """Print a JSON array of all identifier names (keys) for the lockfiles we update"""
    keys = [k for k in LOCKFILES.keys()]
    print(json.dumps(keys, separators=(",", ":")))


def list_skip_branches():
    """List relevant branches that are currently being merged by Bors, as JSON"""
    skip = []

    for name in LOCKFILES.keys():
        branch = branch_name(name)

        try:
            output = sp.check_output(
                [
                    "gh",
                    "pr",
                    "view",
                    branch,
                    "--repo",
                    os.getenv("GITHUB_REPOSITORY", "https://github.com/rust-lang/rust"),
                    "--json",
                    "labels,state",
                ]
            )
        except sp.CalledProcessError:
            # Branch doesn't exist or network error
            continue

        j = json.loads(output)
        is_open = j["state"] == "OPEN"
        waiting_on_bors = any(
            label["name"] == "S-waiting-on-bors" for label in j["labels"]
        )

        # No need to create a new PR if there is already one in the queue
        if is_open and waiting_on_bors:
            skip.append(name)

    print(json.dumps(skip, separators=(",", ":")))


def prepare_pr_files(name: str):
    """Create a commit message and pull request body"""
    with open(UPDATE_JSON_OUTFILE) as f:
        j = json.load(f)

    kw = j[name]
    kw["name"] = name

    commit_text = COMMIT_TEMPLATE.format(**kw)
    pr_body_text = PR_BODY_TEMPLATE.format(**kw)

    Path(COMMIT_TEXT_OUTFILE).write_text(commit_text)
    print(f"Wrote output to {COMMIT_TEXT_OUTFILE}")

    Path(PR_BODY_OUTFILE).write_text(pr_body_text)
    print(f"Wrote output to {PR_BODY_OUTFILE}")


def restore_lockfile(name: str):
    """Update a lockfile path with contents from `update_output.json`"""
    with open(UPDATE_JSON_OUTFILE) as f:
        j = json.load(f)

    path = Path(j[name]["lockfile_path"])
    contents = j[name]["lockfile"]
    path.write_text(contents)

    print(f"Updated lockfile at {path}")


def exit_printing_usage():
    print(USAGE)
    sys.exit(1)


def main():
    if sys.argv[1] == "--run-update":
        if len(sys.argv) != 2:
            exit_printing_usage()
        run_update()

    elif sys.argv[1] == "--print-names":
        if len(sys.argv) != 2:
            exit_printing_usage()
        print_names()

    elif sys.argv[1] == "--list-skip-branches":
        if len(sys.argv) != 2:
            exit_printing_usage()
        list_skip_branches()

    elif sys.argv[1] == "--prepare-pr-files":
        if len(sys.argv) != 3:
            print("expected two arguments")
            exit_printing_usage()
        prepare_pr_files(sys.argv[2])

    elif sys.argv[1] == "--restore-lockfile":
        if len(sys.argv) != 3:
            print("expected two arguments")
            exit_printing_usage()
        restore_lockfile(sys.argv[2])

    else:
        exit_printing_usage()


if __name__ == "__main__":
    main()
