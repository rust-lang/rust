We are currently in the process of discussing Clippy 1.0 via the RFC process in https://github.com/rust-lang/rfcs/pull/2476 . The RFC's goal is to clarify policies around lint categorizations and the policy around which lints should be in the compiler and which lints should be in Clippy. Please leave your thoughts on the RFC PR.

# Clippy

[![Build Status](https://travis-ci.org/rust-lang-nursery/rust-clippy.svg?branch=master)](https://travis-ci.org/rust-lang-nursery/rust-clippy)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/id677xpw1dguo7iw?svg=true)](https://ci.appveyor.com/project/rust-lang-libs/rust-clippy)
[![Current Version](https://meritbadge.herokuapp.com/clippy)](https://crates.io/crates/clippy)
[![License: MPL-2.0](https://img.shields.io/crates/l/clippy.svg)](#license)

A collection of lints to catch common mistakes and improve your [Rust](https://github.com/rust-lang/rust) code.

[There are 273 lints included in this crate!](https://rust-lang-nursery.github.io/rust-clippy/master/index.html)

We have a bunch of lint categories to allow you to choose how much Clippy is supposed to ~~annoy~~ help you:

* `clippy` (everything that has no false positives)
* `clippy_pedantic` (everything)
* `clippy_nursery` (new lints that aren't quite ready yet)
* `clippy_style` (code that should be written in a more idiomatic way)
* `clippy_complexity` (code that does something simple but in a complex way)
* `clippy_perf` (code that can be written in a faster way)
* `clippy_cargo` (checks against the cargo manifest)
* **`clippy_correctness`** (code that is just outright wrong or very very useless)

More to come, please [file an issue](https://github.com/rust-lang-nursery/rust-clippy/issues) if you have ideas!

Table of contents:

*   [Usage instructions](#usage)
*   [Configuration](#configuration)
*   [License](#license)

## Usage

Since this is a tool for helping the developer of a library or application
write better code, it is recommended not to include Clippy as a hard dependency.
Options include using it as an optional dependency, as a cargo subcommand, or
as an included feature during build. These options are detailed below.

### As a cargo subcommand (`cargo clippy`)

One way to use Clippy is by installing Clippy through rustup as a cargo
subcommand.

#### Step 1: Install rustup

You can install [rustup](http://rustup.rs/) on supported platforms. This will help
us install clippy and its dependencies.

If you already have rustup installed, update to ensure you have the latest
rustup and compiler:

```terminal
rustup update
```

#### Step 2: Install nightly toolchain

As a general rule Clippy will only work with the *latest* Rust nightly for now.

To install Rust nightly with [rustup](https://rustup.rs/):

```terminal
rustup install nightly
```

#### Step 3: Install clippy

Once you have rustup and the nightly toolchain installed, run the following command:

```terminal
rustup component add clippy-preview --toolchain=nightly
```

Now you can run Clippy by invoking `cargo +nightly clippy`.

### Running Clippy from the command line without installing it

To have cargo compile your crate with Clippy without Clippy installation
in your code, you can use:

```terminal
cargo run --bin cargo-clippy --manifest-path=path_to_clippys_Cargo.toml
```

*[Note](https://github.com/rust-lang-nursery/rust-clippy/wiki#a-word-of-warning):*
Be sure that Clippy was compiled with the same version of rustc that cargo invokes here!

## Configuration

Some lints can be configured in a TOML file named with `clippy.toml` or `.clippy.toml`. It contains basic `variable = value` mapping eg.

```toml
blacklisted-names = ["toto", "tata", "titi"]
cyclomatic-complexity-threshold = 30
```

See the [list of lints](https://rust-lang-nursery.github.io/rust-clippy/master/index.html) for more information about which lints can be configured and the
meaning of the variables.

To deactivate the “for further information visit *lint-link*” message you can
define the `CLIPPY_DISABLE_DOCS_LINKS` environment variable.

### Allowing/denying lints

You can add options  to `allow`/`warn`/`deny`:

*   the whole set of `Warn` lints using the `clippy` lint group (`#![deny(clippy)]`)

*   all lints using both the `clippy` and `clippy_pedantic` lint groups (`#![deny(clippy)]`,
    `#![deny(clippy_pedantic)]`). Note that `clippy_pedantic` contains some very aggressive
    lints prone to false positives.

*   only some lints (`#![deny(single_match, box_vec)]`, etc)

*   `allow`/`warn`/`deny` can be limited to a single function or module using `#[allow(...)]`, etc

Note: `deny` produces errors instead of warnings.

For convenience, `cargo clippy` automatically defines a `cargo-clippy`
feature. This lets you set lint levels and compile with or without Clippy
transparently:

```rust
#[cfg_attr(feature = "cargo-clippy", allow(needless_lifetimes))]
```

## Updating rustc

Sometimes, rustc moves forward without Clippy catching up. Therefore updating
rustc may leave Clippy a non-functional state until we fix the resulting
breakage.

You can use the [rust-update](rust-update) script to update rustc only if
Clippy would also update correctly.

## License

Licensed under [MPL](https://www.mozilla.org/MPL/2.0/).
If you're having issues with the license, let me know and I'll try to change it to something more permissive.
