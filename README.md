# rust-clippy

[![Build Status](https://travis-ci.org/rust-lang-nursery/rust-clippy.svg?branch=master)](https://travis-ci.org/rust-lang-nursery/rust-clippy)
[![Windows build status](https://ci.appveyor.com/api/projects/status/github/rust-lang-nursery/rust-clippy?svg=true)](https://ci.appveyor.com/project/rust-lang-nursery/rust-clippy)
[![Current Version](http://meritbadge.herokuapp.com/clippy)](https://crates.io/crates/clippy)
[![License: MPL-2.0](https://img.shields.io/crates/l/clippy.svg)](#license)

A collection of lints to catch common mistakes and improve your [Rust](https://github.com/rust-lang/rust) code.

[There are 257 lints included in this crate!](https://rust-lang-nursery.github.io/rust-clippy/master/index.html)

We have a bunch of lint categories to allow you to choose how much clippy is supposed to ~~annoy~~ help you:

* `clippy` (everything that has no false positives)
* `clippy_pedantic` (everything)
* `clippy_nursery` (new lints that aren't quite ready yet)
* `clippy_style` (code that should be written in a more idiomatic way)
* `clippy_complexity` (code that does something simple but in a complex way)
* `clippy_perf` (code that can be written in a faster way)
* **`clippy_correctness`** (code that is just outright wrong or very very useless)

More to come, please [file an issue](https://github.com/rust-lang-nursery/rust-clippy/issues) if you have ideas!

Table of contents:

*   [Usage instructions](#usage)
*   [Configuration](#configuration)
*   [License](#license)

## Usage

Since this is a tool for helping the developer of a library or application
write better code, it is recommended not to include clippy as a hard dependency.
Options include using it as an optional dependency, as a cargo subcommand, or
as an included feature during build. All of these options are detailed below.

As a general rule clippy will only work with the *latest* Rust nightly for now.

To install Rust nightly, the recommended way is to use [rustup](https://rustup.rs/):

```terminal
rustup install nightly
```

### As a cargo subcommand (`cargo clippy`)

One way to use clippy is by installing clippy through cargo as a cargo
subcommand.

```terminal
cargo +nightly install clippy
```

(The `+nightly` is not necessary if your default `rustup` install is nightly)

Now you can run clippy by invoking `cargo +nightly clippy`.

To update the subcommand together with the latest nightly use the [rust-update](rust-update) script or run:

```terminal
rustup update nightly
cargo +nightly install --force clippy
```

In case you are not using rustup, you need to set the environment flag
`SYSROOT` during installation so clippy knows where to find `librustc` and
similar crates.

```terminal
SYSROOT=/path/to/rustc/sysroot cargo install clippy
```

### Optional dependency

In some cases you might want to include clippy in your project directly, as an
optional dependency. To do this, just modify `Cargo.toml`:

```toml
[dependencies]
clippy = { version = "*", optional = true }
```

And, in your `main.rs` or `lib.rs`, add these lines:

```rust
#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
```

Then build by enabling the feature: `cargo +nightly build --features "clippy"`.

Instead of adding the `cfg_attr` attributes you can also run clippy on demand:
`cargo rustc --features clippy -- -Z no-trans -Z extra-plugins=clippy`
(the `-Z no trans`, while not necessary, will stop the compilation process after
typechecking (and lints) have completed, which can significantly reduce the runtime).

Alternatively, to only run clippy when testing:

```toml
[dev-dependencies]
clippy = { version = "*" }
```

and add to `main.rs` or  `lib.rs`:

```
#![cfg_attr(test, feature(plugin))]
#![cfg_attr(test, plugin(clippy))]
```

### Running clippy from the command line without installing it

To have cargo compile your crate with clippy without clippy installation and without needing `#![plugin(clippy)]`
in your code, you can use:

```terminal
cargo run --bin cargo-clippy --manifest-path=path_to_clippys_Cargo.toml
```

*[Note](https://github.com/rust-lang-nursery/rust-clippy/wiki#a-word-of-warning):*
Be sure that clippy was compiled with the same version of rustc that cargo invokes here!

### As a Compiler Plugin

*Note:* This is not a recommended installation method.

Since stable Rust is backwards compatible, you should be able to
compile your stable programs with nightly Rust with clippy plugged in to
circumvent this.

Add in your `Cargo.toml`:

```toml
[dependencies]
clippy = "*"
```

You then need to add `#![feature(plugin)]` and `#![plugin(clippy)]` to the top
of your crate entry point (`main.rs` or `lib.rs`).

Sample `main.rs`:

```rust
#![feature(plugin)]

#![plugin(clippy)]


fn main(){
    let x = Some(1u8);
    match x {
        Some(y) => println!("{:?}", y),
        _ => ()
    }
}
```

Produces this warning:

```terminal
src/main.rs:8:5: 11:6 warning: you seem to be trying to use match for destructuring a single type. Consider using `if let`, #[warn(single_match)] on by default
src/main.rs:8     match x {
src/main.rs:9         Some(y) => println!("{:?}", y),
src/main.rs:10         _ => ()
src/main.rs:11     }
src/main.rs:8:5: 11:6 help: Try
if let Some(y) = x { println!("{:?}", y) }
```

## Configuration

Some lints can be configured in a TOML file named with `clippy.toml` or `.clippy.toml`. It contains basic `variable = value` mapping eg.

```toml
blacklisted-names = ["toto", "tata", "titi"]
cyclomatic-complexity-threshold = 30
```

See the [list of lints](https://rust-lang-nursery.github.io/rust-clippy/master/index.html) for more information about which lints can be configured and the
meaning of the variables.

You can also specify the path to the configuration file with:

```rust
#![plugin(clippy(conf_file="path/to/clippy's/configuration"))]
```

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
feature. This lets you set lint levels and compile with or without clippy
transparently:

```rust
#[cfg_attr(feature = "cargo-clippy", allow(needless_lifetimes))]
```

## Updating rustc

Sometimes, rustc moves forward without clippy catching up. Therefore updating
rustc may leave clippy a non-functional state until we fix the resulting
breakage.

You can use the [rust-update](rust-update) script to update rustc only if
clippy would also update correctly.

## License

Licensed under [MPL](https://www.mozilla.org/MPL/2.0/).
If you're having issues with the license, let me know and I'll try to change it to something more permissive.
