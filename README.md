# rust-clippy

[![Build Status](https://travis-ci.org/rust-lang-nursery/rust-clippy.svg?branch=master)](https://travis-ci.org/rust-lang-nursery/rust-clippy)
[![Windows build status](https://ci.appveyor.com/api/projects/status/github/rust-lang-nursery/rust-clippy?svg=true)](https://ci.appveyor.com/project/rust-lang-nursery/rust-clippy)
[![Current Version](http://meritbadge.herokuapp.com/clippy)](https://crates.io/crates/clippy)
[![License: MPL-2.0](https://img.shields.io/crates/l/clippy.svg)](#License)

A collection of lints to catch common mistakes and improve your Rust code.

Table of contents:

*   [Lint list](#lints)
*   [Usage instructions](#usage)
*   [Configuration](#configuration)
*   [License](#license)

## Usage

Since this is a tool for helping the developer of a library or application
write better code, it is recommended not to include clippy as a hard dependency.
Options include using it as an optional dependency, as a cargo subcommand, or
as an included feature during build. All of these options are detailed below.

As a general rule clippy will only work with the *latest* Rust nightly for now.

### Optional dependency

If you want to make clippy an optional dependency, you can do the following:

In your `Cargo.toml`:

```toml
[dependencies]
clippy = {version = "*", optional = true}

[features]
default = []
```

And, in your `main.rs` or `lib.rs`:

```rust
#![cfg_attr(feature="clippy", feature(plugin))]

#![cfg_attr(feature="clippy", plugin(clippy))]
```

Then build by enabling the feature: `cargo build --features "clippy"`

Instead of adding the `cfg_attr` attributes you can also run clippy on demand:
`cargo rustc --features clippy -- -Z no-trans -Z extra-plugins=clippy`
(the `-Z no trans`, while not necessary, will stop the compilation process after
typechecking (and lints) have completed, which can significantly reduce the runtime).

### As a cargo subcommand (`cargo clippy`)

An alternate way to use clippy is by installing clippy through cargo as a cargo
subcommand.

```terminal
cargo install clippy
```

Now you can run clippy by invoking `cargo clippy`, or
`rustup run nightly cargo clippy` directly from a directory that is usually
compiled with stable.

In case you are not using rustup, you need to set the environment flag
`SYSROOT` during installation so clippy knows where to find `librustc` and
similar crates.

```terminal
SYSROOT=/path/to/rustc/sysroot cargo install clippy
```

### Running clippy from the command line without installing

To have cargo compile your crate with clippy without needing `#![plugin(clippy)]`
in your code, you can use:

```terminal
cargo rustc -- -L /path/to/clippy_so/dir/ -Z extra-plugins=clippy
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

Some lints can be configured in a `clippy.toml` file. It contains basic `variable = value` mapping eg.

```toml
blacklisted-names = ["toto", "tata", "titi"]
cyclomatic-complexity-threshold = 30
```

See the wiki for more information about which lints can be configured and the
meaning of the variables.

You can also specify the path to the configuration file with:

```rust
#![plugin(clippy(conf_file="path/to/clippy's/configuration"))]
```

To deactivate the “for further information visit *wiki-link*” message you can
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
features. This lets you set lints level and compile with or without clippy
transparently:

```rust
#[cfg_attr(feature = "cargo-clippy", allow(needless_lifetimes))]
```

## Lints

[There are 209 lints included in this crate](https://rust-lang-nursery.github.io/rust-clippy/master/index.html)

More to come, please [file an issue](https://github.com/rust-lang-nursery/rust-clippy/issues) if you have ideas!

## License

Licensed under [MPL](https://www.mozilla.org/MPL/2.0/).
If you're having issues with the license, let me know and I'll try to change it to something more permissive.
