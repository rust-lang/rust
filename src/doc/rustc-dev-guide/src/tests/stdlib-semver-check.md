# Standard library semantic versioning breakage check

The `x86_64-gnu-stdlib-semver-check` job runs the [`cargo-semver-checks`][csc] (c-s-c) tool on the standard library (`core`, `alloc` and `std`) in order to find potential unintended semantic versioning (semver) breakages. It does so by analyzing the rustdoc JSON output (from the `rust-docs-json` component) of the parent merge commit, and the current commit being merged. When it runs, one of five things can happen:

1. Everything proceeds correctly, c-s-c does not find any breakage.
2. The rustdoc JSON version was bumped recently, and c-s-c cannot handle it yet. This case will result in the test ending with a success, and printing a warning that c-s-c needs to be updated. Once c-s-c releases a version that supports the new JSON format, it should go to 1. again.
   - We currently install the latest released version of c-s-c in this job, so its version does not need to be updated manually in the `rust-lang/rust` repository.
3. c-s-c detects a breakage, but it is a false positive. In this case, please report the false positive to [this][semver-topic] Zulip channel, and [bump the stamp file](#bumping-the-stdlib-semver-stamp-file). 
4. c-s-c detects a real breakage, and it helped you find unintended beakage. Yay! In this case, please consider reporting the success to [this][semver-topic] Zulip channel.
5. c-s-c detects a real breakage, but you want to land it anyway (maybe it is an edge case that was FCPed). In that case, [bump the stamp file](#bumping-the-stdlib-semver-stamp-file). 

## Bumping the stdlib semver stamp file

If you want to let CI pass on a PR where c-s-c detects breakage (whether it is real or not), you have to modify the `src/bootstrap/stdlib-semver-check-stamp` file. Please update the PR number in which you modify this file at the bottom of the file. This will ensure that the test will stay green, regardless of what c-s-c detects.

## Running the check manually

You can manually run the semver check locally using `./x test std-semver-check --set rust.stdlib-semver-baseline=${PARENT}`, where `PARENT` is a commit SHA against which you want to compare the in-tree stdlib. If you do not specify it, bootstrap will select the latest upstream commit that it finds in your local git history.

[semver-topic]: https://rust-lang.zulipchat.com/#narrow/channel/219381-t-libs/topic/Breakages.20detected.20by.20cargo-semver-checks/with/615570111
[csc]: https://github.com/obi1kenobi/cargo-semver-checks
