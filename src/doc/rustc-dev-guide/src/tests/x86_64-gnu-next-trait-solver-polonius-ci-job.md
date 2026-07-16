# Pre-stabilization testing of the next solver and polonius alpha on CI

We want to improve the continuous test coverage of next solver and polonius
alpha with a goal to stabilize them in 2026. To achieve this, we need to ensure
these unstable features are well tested and continuously well tested so they do
not regress.

This calls for a dedicated CI job [`x86_64-gnu-next-trait-solver-polonius`] which is
blocking in PR CI and thus also Merge CI to catch regressions.

For more context, see:

* [MCP 996: Test new solver and polonius alpha on CI](https://github.com/rust-lang/compiler-team/issues/996)
* Tracking issue: <https://github.com/rust-lang/rust/issues/157780>

## What is tested

* For the next solver:
    * We want to ensure we can build the standard library with the next solver.
    * We want to ensure we can bootstrap a full toolchain with the next solver.
    * We do both by building the stage 2 library with the next solver enabled at
      stage 1.
* For the polonius alpha:
    * We run the UI test suite under the polonius [compare
      mode](./compiletest.md#compare-modes), against the stage 1 compiler/std
      built with the next solver enabled.

> **Note**
>
> To reproduce failures locally:
>
> ```console
> $ RUSTFLAGS_NOT_BOOTSTRAP="-Znext-solver=globally" ./x build library --stage 2
> ```
>
> ```console
> $ RUSTFLAGS_NOT_BOOTSTRAP="-Znext-solver=globally" ./x test tests/ui --compare-mode polonius --stage 1
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


[`x86_64-gnu-next-trait-solver-polonius`]: https://github.com/rust-lang/rust/pull/157322
