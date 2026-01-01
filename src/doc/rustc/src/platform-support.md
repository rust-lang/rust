# Platform Support

<style type="text/css">
    td code {
        white-space: nowrap;
    }
</style>

Support for different platforms ("targets") are organized into three tiers,
each with a different set of guarantees. For more information on the policies
for targets at each tier, see the [Target Tier Policy](target-tier-policy.md).

Targets are identified by their "target triple" which is the string to inform
the compiler what kind of output should be produced.

Component availability is tracked [here](https://rust-lang.github.io/rustup-components-history/).

## Tier 1 with Host Tools

Tier 1 targets can be thought of as "guaranteed to work". The Rust project
builds official binary releases for each tier 1 target, and automated testing
ensures that each tier 1 target builds and passes tests after each change.

Tier 1 targets with host tools additionally support running tools like `rustc`
and `cargo` natively on the target, and automated testing ensures that tests
pass for the host tools as well. This allows the target to be used as a
development platform, not just a compilation target. For the full requirements,
see [Tier 1 with Host Tools](target-tier-policy.md#tier-1-with-host-tools) in
the Target Tier Policy.

All tier 1 targets with host tools support the full standard library.

{{TIER_1_HOST_TABLE}}

[^x86_32-floats-return-ABI]: Due to limitations of the C ABI, floating-point support on `i686` targets is non-compliant: floating-point return values are passed via an x87 register, so NaN payload bits can be lost. Functions with the default Rust ABI are not affected. See [issue #115567][x86-32-float-return-issue].

[^win32-msvc-alignment]: Due to non-standard behavior of MSVC, native C code on this target can cause types with an alignment of more than 4 bytes to be incorrectly aligned to only 4 bytes (this affects, e.g., `u64` and `i64`). Rust applies some mitigations to reduce the impact of this issue, but this can still cause unsoundness due to unsafe code that (correctly) assumes that references are always properly aligned. See [issue #112480](https://github.com/rust-lang/rust/issues/112480).

[77071]: https://github.com/rust-lang/rust/issues/77071
[x86-32-float-return-issue]: https://github.com/rust-lang/rust/issues/115567

## Tier 1

Tier 1 targets can be thought of as "guaranteed to work". The Rust project
builds official binary releases for each tier 1 target, and automated testing
ensures that each tier 1 target builds and passes tests after each change. For
the full requirements, see [Tier 1 target
policy](target-tier-policy.md#tier-1-target-policy) in the Target Tier Policy.

{{TIER_1_NOHOST_TABLE}}

## Tier 2 with Host Tools

Tier 2 targets can be thought of as "guaranteed to build". The Rust project
builds official binary releases of the standard library (or, in some cases,
only the `core` library) for each tier 2 target, and automated builds
ensure that each tier 2 target can be used as build target after each change. Automated tests are
not always run so it's not guaranteed to produce a working build, but tier 2
targets often work to quite a good degree and patches are always welcome!

Tier 2 target-specific code is not closely scrutinized by Rust team(s) when
modifications are made. Bugs are possible in all code, but the level of quality
control for these targets is likely to be lower. See [library team
policy](https://std-dev-guide.rust-lang.org/policy/target-code.html) for
details on the review practices for standard library code.

Tier 2 targets with host tools additionally support running tools like `rustc`
and `cargo` natively on the target, and automated builds ensure that the host
tools build as well. This allows the target to be used as a development
platform, not just a compilation target. For the full requirements, see [Tier 2
with Host Tools](target-tier-policy.md#tier-2-with-host-tools) in the Target
Tier Policy.

All tier 2 targets with host tools support the full standard library.

**NOTE:** The `rust-docs` component is not usually built for tier 2 targets,
so Rustup may install the documentation for a similar tier 1 target instead.

{{TIER_2_HOST_TABLE}}

## Tier 2 without Host Tools

Tier 2 targets can be thought of as "guaranteed to build". The Rust project
builds official binary releases of the standard library (or, in some cases,
only the `core` library) for each tier 2 target, and automated builds
ensure that each tier 2 target can be used as build target after each change. Automated tests are
not always run so it's not guaranteed to produce a working build, but tier 2
targets often work to quite a good degree and patches are always welcome! For
the full requirements, see [Tier 2 target
policy](target-tier-policy.md#tier-2-target-policy) in the Target Tier Policy.

The `std` column in the table below has the following meanings:

* ✓ indicates the full standard library is available.
* \* indicates the target only supports [`no_std`] development.
* ? indicates the standard library support is a work-in-progress.

[`no_std`]: https://rust-embedded.github.io/book/intro/no-std.html

Tier 2 target-specific code is not closely scrutinized by Rust team(s) when
modifications are made. Bugs are possible in all code, but the level of quality
control for these targets is likely to be lower. See [library team
policy](https://std-dev-guide.rust-lang.org/policy/target-code.html) for
details on the review practices for standard library code.

**NOTE:** The `rust-docs` component is not usually built for tier 2 targets,
so Rustup may install the documentation for a similar tier 1 target instead.

{{TIER_2_NOHOST_TABLE}}

[^x86_32-floats-x87]: Floating-point support on `i586` targets is non-compliant: the `x87` registers and instructions used for these targets do not provide IEEE-754-compliant behavior, in particular when it comes to rounding and NaN payload bits. See [issue #114479][x86-32-float-issue].

[x86-32-float-issue]: https://github.com/rust-lang/rust/issues/114479

[wasi-rename]: https://github.com/rust-lang/compiler-team/issues/607

[Fortanix ABI]: https://edp.fortanix.com/

## Tier 3

Tier 3 targets are those which the Rust codebase has support for, but which the
Rust project does not build or test automatically, so they may or may not work.
Official builds are not available. For the full requirements, see [Tier 3
target policy](target-tier-policy.md#tier-3-target-policy) in the Target Tier
Policy.

The `std` column in the table below has the following meanings:

* ✓ indicates the full standard library is available.
* \* indicates the target only supports [`no_std`] development.
* ? indicates the standard library support is unknown or a work-in-progress.

[`no_std`]: https://rust-embedded.github.io/book/intro/no-std.html

Tier 3 target-specific code is not closely scrutinized by Rust team(s) when
modifications are made. Bugs are possible in all code, but the level of quality
control for these targets is likely to be lower. See [library team
policy](https://std-dev-guide.rust-lang.org/policy/target-code.html) for
details on the review practices for standard library code.

The `host` column indicates whether the codebase includes support for building
host tools.

{{TIER_3_TABLE}}

[runs on NVIDIA GPUs]: https://github.com/japaric-archived/nvptx#targets
[the AMD GPU]: https://llvm.org/docs/AMDGPUUsage.html#processors
