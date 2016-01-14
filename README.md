# rustfmt [![Build Status](https://travis-ci.org/rust-lang-nursery/rustfmt.svg)](https://travis-ci.org/rust-lang-nursery/rustfmt)

A tool for formatting Rust code according to style guidelines.

If you'd like to help out (and you should, it's a fun project!), see
[Contributing.md](Contributing.md).

## Quick start

To install:

```
cargo install rustfmt
```

to run on a cargo project in the current working directory:

```
cargo fmt
```

## Installation

> **Note:** this method currently requires you to be running cargo 0.6.0 or
> newer.

```
cargo install rustfmt
```

or if you're using [`multirust`](https://github.com/brson/multirust)

```
multirust run nightly cargo install rustfmt
```

Usually cargo-fmt, which enables usage of Cargo subcommand `cargo fmt`, is
installed alongside rustfmt. To only install rustfmt run

```
cargo install --no-default-features rustfmt
```

## Running

You can run Rustfmt by just typing `rustfmt filename` if you used `cargo
install`. This runs rustfmt on the given file, if the file includes out of line
modules, then we reformat those too. So to run on a whole module or crate, you
just need to run on the root file (usually mod.rs or lib.rs). Rustfmt can also
read data from stdin. Alternatively, you can use `cargo fmt` to format all
binary and library targets of your crate.

You'll probably want to specify the write mode. Currently, there are modes for
replace, overwrite, display, and coverage. The replace mode is the default
and overwrites the original files after renaming them. In overwrite mode,
rustfmt does not backup the source files. To print the output to stdout, use the
display mode. The write mode can be set by passing the `--write-mode` flag on
the command line.

`rustfmt filename --write-mode=display` prints the output of rustfmt to the
screen, for example.

You can run `rustfmt --help` for more information.

`cargo fmt` uses `--write-mode=replace` by default.


## Running Rustfmt from your editor

* [Vim](http://johannh.me/blog/rustfmt-vim.html)
* [Emacs](https://github.com/fbergroth/emacs-rustfmt)
* [Sublime Text 3](https://packagecontrol.io/packages/BeautifyRust)
* [Atom](atom.md)
* Visual Studio Code using [RustyCode](https://github.com/saviorisdead/RustyCode) or [vsc-rustfmt](https://github.com/Connorcpu/vsc-rustfmt)

## How to build and test

`cargo build` to build.

`cargo test` to run all tests.

To run rustfmt after this, use `cargo run --bin rustfmt -- filename`. See the
notes above on running rustfmt.


## Configuring Rustfmt

Rustfmt is designed to be very configurable. You can create a TOML file called
rustfmt.toml, place it in the project directory and it will apply the options
in that file. See `cargo run -- --config-help` for the options which are available,
or if you prefer to see source code, [src/config.rs](src/config.rs).

By default, Rustfmt uses a style which (mostly) conforms to the
[Rust style guidelines](https://github.com/rust-lang/rust/tree/master/src/doc/style).
There are many details which the style guidelines do not cover, and in these
cases we try to adhere to a style similar to that used in the
[Rust repo](https://github.com/rust-lang/rust). Once Rustfmt is more complete, and
able to re-format large repositories like Rust, we intend to go through the Rust
RFC process to nail down the default style in detail.

If there are styling choices you don't agree with, we are usually happy to add
options covering different styles. File an issue, or even better, submit a PR.


## Gotchas

* For things you do not want rustfmt to mangle, use one of

    ```rust
    #[rustfmt_skip]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    ```
* When you run rustfmt, place a file named rustfmt.toml in target file
  directory or its parents to override the default settings of rustfmt.
* After successful compilation, a `rustfmt` executable can be found in the
  target directory.
