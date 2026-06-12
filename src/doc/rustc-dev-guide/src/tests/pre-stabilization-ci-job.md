# Pre-stabilization testing of new solver and polonius alpha on CI

We want to improve the continuous test coverage of next solver and polonius
alpha with a goal to stabilize them in 2026. To achieve this, we need to ensure
these unstable features are well tested and continuously well tested so they do
not regress.

This calls for a dedicated CI job [`x86_64-gnu-pre-stabilization`] which is
blocking in PR CI and thus also Merge CI to catch regressions.

For more context, see:

* [MCP 996: Test new solver and polonius alpha on CI](https://github.com/rust-lang/compiler-team/issues/996)
* Tracking issue: <https://github.com/rust-lang/rust/issues/157780>

## What is tested

* For next solver:
    * We want to ensure that we can build the stage 2 compiler and standard
      library with next solver.
    * (Later) we also want to ensure that we can bootstrap a full toolchain with
      the next solver.
* For polonius alpha:
    * We wil run the UI test suite under polonius [compare
      mode](./compiletest.md#compare-modes), against the stage 2 compiler/std
      built with next solver as aforementioned.

> **Note**
>
> Initially, we may test only up to `--stage=1`, but the plan is to expand up to
> stage 2.
>
> To reproduce failures locally:
>
> ```console
> $ RUSTFLAGS_NOT_BOOTSTRAP="-Znext-solver=globally" ../x build library --stage 1
> ```
>
> ```console
> $ RUSTFLAGS_NOT_BOOTSTRAP="-Znext-solver=globally" ../x test tests/ui --compare-mode polonius --stage 1
> ```

## What failures and remedies might be specific to this combination?

* This may yield new errors when using unstable features in the standard library
  or the compiler.
* This may sometimes require blessing new tests, or in rare cases, adding test
  annotations or revisions.

## Point of contact

If you have questions or need advice on a test failure in this job, please open
a new topic in [t-compiler zulip
channel](https://rust-lang.zulipchat.com/#narrow/channel/131828-t-compiler).


[`x86_64-gnu-pre-stabilization`]: https://github.com/rust-lang/rust/pull/157322
