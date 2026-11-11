# Cargo Specifics - Checking Conditional Configurations

<!--
This page is currently (as of May 2024) the canonical place for describing the interaction
between Cargo and --check-cfg. It is placed in the rustc book rather than the Cargo book
since check-cfg is primarily a Rust/rustc feature and is therefore considered by T-cargo to
be an implementation detail, at least --check-cfg and the unexpected_cfgs are owned by
rustc, not Cargo.
-->

This document is intended to summarize the principal ways Cargo interacts with
the `unexpected_cfgs` lint and `--check-cfg` flag.
For individual details, refer to the [`--check-cfg` documentation](../check-cfg.md) and
to the [Cargo book](../../cargo/index.html).

> The full list of well known cfgs (aka builtins) can be found under [Checking conditional configurations / Well known names and values](../check-cfg.md#well-known-names-and-values).

## Cargo feature

*See the [`[features]` section in the Cargo book][cargo-features] for more details.*

With the `[features]` table, Cargo provides a mechanism to express conditional compilation and
optional dependencies. Cargo *automatically* declares corresponding cfgs for every feature as
expected.

`Cargo.toml`:
```toml
[features]
serde = ["dep:serde"]
my_feature = []
```

[cargo-features]: ../../cargo/reference/features.html

## `check-cfg` in `[lints.rust]` table

<!-- Note that T-Cargo considers `lints.rust.unexpected_cfgs.check-cfg` to be an
implementation detail and is therefore documented here and not in Cargo. -->

*See the [`[lints]` section in the Cargo book][cargo-lints-table] for more details.*

When using a statically known custom config (i.e., not dependent on a build-script), Cargo provides
the custom lint config `check-cfg` under `[lints.rust.unexpected_cfgs]`.

It can be used to set custom static [`--check-cfg`](../check-cfg.md) args, it is mainly useful when
the list of expected cfgs is known in advance.

`Cargo.toml`:
```toml
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(has_foo)'] }
```

[cargo-lints-table]: ../../cargo/reference/manifest.html#the-lints-section

## `cargo::rustc-check-cfg` for `build.rs`/build-script

*See the [`cargo::rustc-check-cfg` section in the Cargo book][cargo-rustc-check-cfg] for more details.*

When setting a custom config with [`cargo::rustc-cfg`][cargo-rustc-cfg], Cargo provides the
corollary instruction: [`cargo::rustc-check-cfg`][cargo-rustc-check-cfg] to expect custom configs.

`build.rs`:
```rust,ignore (cannot-test-this-because-has_foo-isnt-declared)
fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_foo)");
    //        ^^^^^^^^^^^^^^^^^^^^^^ new with Cargo 1.80
    if has_foo() {
        println!("cargo::rustc-cfg=has_foo");
    }
}
```

[cargo-rustc-cfg]: ../../cargo/reference/build-scripts.html#rustc-cfg
[cargo-rustc-check-cfg]: ../../cargo/reference/build-scripts.html#rustc-check-cfg
