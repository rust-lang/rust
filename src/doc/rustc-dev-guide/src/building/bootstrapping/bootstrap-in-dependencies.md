# `cfg(bootstrap)` in compiler dependencies

The rust compiler uses some external crates that can run into cyclic dependencies with the compiler itself: the compiler needs an updated crate to build, but the crate needs an updated compiler. This page describes how `#[cfg(bootstrap)]` can be used to break this cycle.

## Enabling `#[cfg(bootstrap)]`

Usually the use of `#[cfg(bootstrap)]` in an external crate causes a warning:

```
warning: unexpected `cfg` condition name: `bootstrap`
 --> src/main.rs:1:7
  |
1 | #[cfg(bootstrap)]
  |       ^^^^^^^^^
  |
  = help: expected names are: `docsrs`, `feature`, and `test` and 31 more
  = help: consider using a Cargo feature instead
  = help: or consider adding in `Cargo.toml` the `check-cfg` lint config for the lint:
           [lints.rust]
           unexpected_cfgs = { level = "warn", check-cfg = ['cfg(bootstrap)'] }
  = help: or consider adding `println!("cargo::rustc-check-cfg=cfg(bootstrap)");` to the top of the `build.rs`
  = note: see <https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html> for more information about checking conditional configuration
  = note: `#[warn(unexpected_cfgs)]` on by default
```

This warning can be silenced by adding these lines to the project's `Cargo.toml`:

```toml
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(bootstrap)'] }
```

Now `#[cfg(bootstrap)]` can be used in the crate just like it can be in the compiler: when the bootstrap compiler is used, code annotated with `#[cfg(bootstrap)]` is compiled, otherwise code annotated with `#[cfg(not(bootstrap))]` is compiled.

## The update dance

As a concrete example we'll use a change where the `#[naked]` attribute was made into an unsafe attribute, which caused a cyclic dependency with the `compiler-builtins` crate.

### Step 1: accept the new behavior in the compiler ([#139797](https://github.com/rust-lang/rust/pull/139797))

In this example it is possible to accept both the old and new behavior at the same time by disabling an error.

### Step 2: update the crate ([#821](https://github.com/rust-lang/compiler-builtins/pull/821))

Now in the crate, use `#[cfg(bootstrap)]` to use the old behavior, or `#[cfg(not(bootstrap))]` to use the new behavior.

### Step 3: update the crate version used by the compiler ([#139934](https://github.com/rust-lang/rust/pull/139934))

For `compiler-builtins` this meant a version bump, in other cases it may be a git submodule update.

### Step 4: remove the old behavior from the compiler ([#139753](https://github.com/rust-lang/rust/pull/139753))

The updated crate can now be used. In this example that meant that the old behavior could be removed.
