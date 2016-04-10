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
## Installing from source

To install from source, first checkout to the tag or branch you want to install, then issue
```
cargo install --path  .
```
This will install `rustfmt` in your `~/.cargo/bin`. Make sure to add `~/cargo/bin` directory to 
your PATH variable.

## Running

You can run Rustfmt by just typing `rustfmt filename` if you used `cargo
install`. This runs rustfmt on the given file, if the file includes out of line
modules, then we reformat those too. So to run on a whole module or crate, you
just need to run on the root file (usually mod.rs or lib.rs). Rustfmt can also
read data from stdin. Alternatively, you can use `cargo fmt` to format all
binary and library targets of your crate.

You'll probably want to specify the write mode. Currently, there are modes for
diff, replace, overwrite, display, coverage, and checkstyle.

* `replace` Is the default and overwrites the original files after creating backups of the files.
* `overwrite` Overwrites the original files _without_ creating backups.
* `display` Will print the formatted files to stdout.
* `diff` Will print a diff between the original files and formatted files to stdout.
* `checkstyle` Will output the lines that need to be corrected as a checkstyle XML file,
  that can be used by tools like Jenkins.

The write mode can be set by passing the `--write-mode` flag on
the command line. For example `rustfmt --write-mode=display src/filename.rs`

`cargo fmt` uses `--write-mode=replace` by default.

If you want to restrict reformatting to specific sets of lines, you can
use the `--file-lines` option. Its argument is a JSON array of objects
with `file` and `range` properties, where `file` is a file name, and
`range` is an array representing a range of lines like `[7,13]`. Ranges
are inclusive of both end points. Specifying an empty array will result in
no files being formatted. For example,

```
rustfmt --file-lines '[
    {"file":"src/lib.rs","range":[7,13]},
    {"file":"src/lib.rs","range":[21,29]},
    {"file":"src/foo.rs","range":[10,11]},
    {"file":"src/foo.rs","range":[15,15]}]'
```

would format lines `7-13` and `21-29` of `src/lib.rs`, and lines `10-11`,
and `15` of `src/foo.rs`. No other files would be formatted, even if they
are included as out of line modules from `src/lib.rs`.

If `rustfmt` successfully reformatted the code it will exit with `0` exit
status. Exit status `1` signals some unexpected error, like an unknown option or
a failure to read a file. Exit status `2` is returned if there are syntax errors
in the input files. `rustfmt` can't format syntatically invalid code. Finally,
exit status `3` is returned if there are some issues which can't be resolved
automatically. For example, if you have a very long comment line `rustfmt`
doesn't split it. Instead it prints a warning and exits with `3`.

You can run `rustfmt --help` for more information.


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
in that file. See `rustfmt --config-help` for the options which are available,
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


## Tips

* For things you do not want rustfmt to mangle, use one of

    ```rust
    #[rustfmt_skip]  // requires nightly and #![feature(custom_attribute)] in crate root
    #[cfg_attr(rustfmt, rustfmt_skip)]  // works in stable
    ```
* When you run rustfmt, place a file named rustfmt.toml in target file
  directory or its parents to override the default settings of rustfmt.
* After successful compilation, a `rustfmt` executable can be found in the
  target directory.
* If you're having issues compiling Rustfmt (or compile errors when trying to
  install), make sure you have the most recent version of Rust installed.


## License

Rustfmt is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.
