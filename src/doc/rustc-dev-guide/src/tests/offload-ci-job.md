# Offload CI job

The [`optional-x86_64-gnu-offload`] job provides continuous test coverage for
the experimental `offload` feature.
It is an optional [auto job](./ci.md#auto-builds), so a failure does not prevent a
pull request from being merged.

For more context about the feature, see the [offload tracking issue] and the
[offload internals](../offload/internals.md) chapter.

## What is tested

The job checks:

- host-side LLVM IR generation;
- device-side LLVM IR generation for AMDGPU and NVPTX targets;
- argument lowering in host and device code;

## Running the job

To run the job in a try build, comment on a pull request:

```text
@bors try jobs=optional-x86_64-gnu-offload
```

To run the job locally, run this command from a Rust checkout:

```console
$ cargo run --manifest-path src/ci/citool/Cargo.toml run-local optional-x86_64-gnu-offload
```

See [Testing with Docker](./docker.md) for more information about running CI jobs locally.

## Point of contact

If you have questions or need help with a failure in this job, open a new topic
in the [offload Zulip channel].

[offload Zulip channel]: https://rust-lang.zulipchat.com/#narrow/channel/422870-t-compiler.2Fgpgpu-backend
[offload tracking issue]: https://github.com/rust-lang/rust/issues/131513
[`optional-x86_64-gnu-offload`]: https://github.com/rust-lang/rust/blob/HEAD/src/ci/github-actions/jobs.yml
