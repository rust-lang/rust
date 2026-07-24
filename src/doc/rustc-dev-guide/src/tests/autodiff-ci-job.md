# Autodiff CI job

The [`optional-x86_64-gnu-autodiff`] job provides continuous test coverage for
the experimental `autodiff` feature and its integration with LLVM Enzyme. It is
an optional [auto job](./ci.md#auto-builds), so a failure does not prevent a
pull request from being merged.

For more context about the feature, see the [autodiff tracking issue] and the
[autodiff internals](../autodiff/internals.md) chapter.

## What is tested

The job checks:

- forward- and reverse-mode macro expansion and diagnostics;
- LLVM IR generation for autodiff;
- enforcement of the `autodiff` feature gate.

## Running the job

To run the job in a try build, comment on a pull request:

```text
@bors try jobs=optional-x86_64-gnu-autodiff
```

To run the job locally, run this command from a Rust checkout:

```console
$ cargo run --manifest-path src/ci/citool/Cargo.toml run-local optional-x86_64-gnu-autodiff
```

See [Testing with Docker](./docker.md) for more information about running CI
jobs locally.

## Point of contact

If you have questions or need help with a failure in this job, open a new topic
in the [autodiff Zulip channel]. For suspected Enzyme backend failures, see the
[autodiff debugging guide].

[autodiff Zulip channel]: https://rust-lang.zulipchat.com/#narrow/channel/390790-wg-autodiff
[autodiff debugging guide]: ../autodiff/debugging.md
[autodiff tracking issue]: https://github.com/rust-lang/rust/issues/124509
[`optional-x86_64-gnu-autodiff`]: https://github.com/rust-lang/rust/blob/HEAD/src/ci/github-actions/jobs.yml
