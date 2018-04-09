# rustfmt [![Build Status](https://travis-ci.org/rust-lang-nursery/rustfmt.svg)](https://travis-ci.org/rust-lang-nursery/rustfmt) [![Build Status](https://ci.appveyor.com/api/projects/status/github/rust-lang-nursery/rustfmt?svg=true)](https://ci.appveyor.com/project/nrc/rustfmt) [![crates.io](https://img.shields.io/crates/v/rustfmt-nightly.svg)](https://crates.io/crates/rustfmt-nightly) [![Travis Configuration Status](https://img.shields.io/travis/davidalber/rustfmt-travis.svg?label=travis%20example)](https://travis-ci.org/davidalber/rustfmt-travis)

A tool for formatting Rust code according to style guidelines.

If you'd like to help out (and you should, it's a fun project!), see
[Contributing.md](Contributing.md) and our [Code of
Conduct](CODE_OF_CONDUCT.md).

We are changing the default style used by rustfmt. There is an ongoing [RFC
process][fmt rfcs]. The last version using the old style was 0.8.6. From 0.9
onwards, the RFC style is the default. If you want the old style back, you can
use [legacy-rustfmt.toml](legacy-rustfmt.toml) as your rustfmt.toml.

The current `master` branch uses libsyntax (part of the compiler). It is
published as `rustfmt-nightly`. The `syntex` branch uses Syntex instead of
libsyntax, it is published (for now) as `rustfmt`. Most development happens on
the `master` branch, however, this only supports nightly toolchains. If you use
stable or beta Rust toolchains, you must use the Syntex version (which is likely
to be a bit out of date). Version 0.1 of rustfmt-nightly is forked from version
0.9 of the syntex branch.

You can use rustfmt in Travis CI builds. We provide a minimal Travis CI
configuration (see [here](#checking-style-on-a-ci-server)) and verify its status
using another repository. The status of that repository's build is reported by
the "travis example" badge above.


## Quick start

You can use `rustfmt` on Rust 1.24 and above.

To install:

```
rustup component add rustfmt-preview
```

to run on a cargo project in the current working directory:

```
cargo fmt
```

For the latest and greatest `rustfmt` (nightly required):
```
rustup component add rustfmt-preview --toolchain nightly
```
To run:
```
cargo +nightly fmt
```

## Limitations

Rustfmt tries to work on as much Rust code as possible, sometimes, the code
doesn't even need to compile! As we approach a 1.0 release we are also looking
to limit areas of instability; in particular, post-1.0, the formatting of most
code should not change as Rustfmt improves. However, there are some things that
Rustfmt can't do or can't do well (and thus where formatting might change
significantly, even post-1.0). We would like to reduce the list of limitations
over time.

The following list enumerates areas where Rustfmt does not work or where the
stability guarantees do not apply (we don't make a distinction between the two
because in the future Rustfmt might work on code where it currently does not):

* a program where any part of the program does not parse (parsing is an early
  stage of compilation and in Rust includes macro expansion).
* Macro declarations and uses (current status: some macro declarations and uses
  are formatted).
* Comments, including any AST node with a comment 'inside' (Rustfmt does not
  currently attempt to format comments, it does format code with comments inside, but that formatting may change in the future).
* Rust code in code blocks in comments.
* Any fragment of a program (i.e., stability guarantees only apply to whole
  programs, even where fragments of a program can be formatted today).
* Code containing non-ascii unicode characters (we believe Rustfmt mostly works
  here, but do not have the test coverage or experience to be 100% sure).
* Bugs in Rustfmt (like any software, Rustfmt has bugs, we do not consider bug
  fixes to break our stability guarantees).


## Installation

```
rustup component add rustfmt-preview
```

## Installing from source

To install from source (nightly required), first checkout to the tag or branch you want to install, then issue
```
cargo install --path .
```

This will install `rustfmt` in your `~/.cargo/bin`. Make sure to add `~/.cargo/bin` directory to
your PATH variable.


## Running

You can run Rustfmt by just typing `rustfmt filename` if you used `cargo
install`. This runs rustfmt on the given file, if the file includes out of line
modules, then we reformat those too. So to run on a whole module or crate, you
just need to run on the root file (usually mod.rs or lib.rs). Rustfmt can also
read data from stdin. Alternatively, you can use `cargo fmt` to format all
binary and library targets of your crate.

You'll probably want to specify the write mode. Currently, there are modes for
`diff`, `replace`, `overwrite`, `display`, `coverage`, `checkstyle`, and `plain`.

* `overwrite` Is the default and overwrites the original files _without_ creating backups.
* `replace` Overwrites the original files after creating backups of the files.
* `display` Will print the formatted files to stdout.
* `plain` Also writes to stdout, but with no metadata.
* `diff` Will print a diff between the original files and formatted files to stdout.
         Will also exit with an error code if there are any differences.
* `checkstyle` Will output the lines that need to be corrected as a checkstyle XML file,
  that can be used by tools like Jenkins.

The write mode can be set by passing the `--write-mode` flag on
the command line. For example `rustfmt --write-mode=display src/filename.rs`

`cargo fmt` uses `--write-mode=overwrite` by default.

If you want to restrict reformatting to specific sets of lines, you can
use the `--file-lines` option. Its argument is a JSON array of objects
with `file` and `range` properties, where `file` is a file name, and
`range` is an array representing a range of lines like `[7,13]`. Ranges
are 1-based and inclusive of both end points. Specifying an empty array
will result in no files being formatted. For example,

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
in the input files. `rustfmt` can't format syntactically invalid code. Finally,
exit status `3` is returned if there are some issues which can't be resolved
automatically. For example, if you have a very long comment line `rustfmt`
doesn't split it. Instead it prints a warning and exits with `3`.

You can run `rustfmt --help` for more information.


## Running Rustfmt from your editor

* [Vim](https://github.com/rust-lang/rust.vim#formatting-with-rustfmt)
* [Emacs](https://github.com/rust-lang/rust-mode)
* [Sublime Text 3](https://packagecontrol.io/packages/RustFmt)
* [Atom](atom.md)
* Visual Studio Code using [vscode-rust](https://github.com/editor-rs/vscode-rust), [vsc-rustfmt](https://github.com/Connorcpu/vsc-rustfmt) or [rls_vscode](https://github.com/jonathandturner/rls_vscode) through RLS.

## Checking style on a CI server

To keep your code base consistently formatted, it can be helpful to fail the CI build
when a pull request contains unformatted code. Using `--write-mode=diff` instructs
rustfmt to exit with an error code if the input is not formatted correctly.
It will also print any found differences.

A minimal Travis setup could look like this (requires Rust 1.24.0 or greater):

```yaml
language: rust
before_script:
- rustup component add rustfmt-preview
script:
- cargo fmt --all -- --write-mode=diff
- cargo build
- cargo test
```

## How to build and test

`cargo build` to build.

`cargo test` to run all tests.

To run rustfmt after this, use `cargo run --bin rustfmt -- filename`. See the
notes above on running rustfmt.


## Configuring Rustfmt

Rustfmt is designed to be very configurable. You can create a TOML file called
`rustfmt.toml` or `.rustfmt.toml`, place it in the project or any other parent
directory and it will apply the options in that file. See `rustfmt
--config-help` for the options which are available, or if you prefer to see
visual style previews, [Configurations.md](Configurations.md).

By default, Rustfmt uses a style which conforms to the [Rust style guide][style
guide] that has been formalized through the [style RFC
process][fmt rfcs].

Configuration options are either stable or unstable. Stable options can always
be used, while unstable ones are only available on a nightly toolchain, and opt-in.
See [Configurations.md](Configurations.md) for details.


## Tips

* For things you do not want rustfmt to mangle, use one of

    ```rust
    #[rustfmt_skip]  // requires nightly and #![feature(custom_attribute)] in crate root
    #[cfg_attr(rustfmt, rustfmt_skip)]  // works in stable
    ```
* When you run rustfmt, place a file named `rustfmt.toml` or `.rustfmt.toml` in
  target file directory or its parents to override the default settings of
  rustfmt. You can generate a file containing the default configuration with
  `rustfmt --dump-default-config rustfmt.toml` and customize as needed.
* After successful compilation, a `rustfmt` executable can be found in the
  target directory.
* If you're having issues compiling Rustfmt (or compile errors when trying to
  install), make sure you have the most recent version of Rust installed.

* If you get an error like `error while loading shared libraries` while starting
  up rustfmt you should try the following:

  On Linux:

  ```
  export LD_LIBRARY_PATH=$(rustc --print sysroot)/lib:$LD_LIBRARY_PATH
  ```

  On MacOS:

  ```
  export DYLD_LIBRARY_PATH=$(rustc --print sysroot)/lib:$DYLD_LIBRARY_PATH
  ```

  On Windows (Git Bash/Mingw):

  ```
  export PATH=$(rustc --print sysroot)/lib/rustlib/x86_64-pc-windows-gnu/lib/:$PATH
  ```

  (Substitute `x86_64` by `i686` and `gnu` by `msvc` depending on which version of rustc was used to install rustfmt).

## License

Rustfmt is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.

[rust]: https://github.com/rust-lang/rust
[fmt rfcs]: https://github.com/rust-lang-nursery/fmt-rfcs
[style guide]: https://github.com/rust-lang-nursery/fmt-rfcs/blob/master/guide/guide.md
