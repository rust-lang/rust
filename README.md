# rustfmt

A tool for formatting Rust code according to style guidelines.


## Installation

> **Note:** this method currently requires you to be running a nightly install
> of Rust as `cargo install` has not yet made its way onto the stable channel.

```
cargo install --git https://github.com/nrc/rustfmt
```

or if you're using [`multirust`](https://github.com/brson/multirust)

```
multirust run nightly cargo install --git https://github.com/nrc/rustfmt
```


## Running Rustfmt from Vim

See [instructions](http://johannh.me/blog/rustfmt-vim.html).


## How to build and test

First make sure you've got Rust **1.4.0** or greater available, then:

`cargo build` to build.

`cargo test` to run all tests.

`cargo run -- filename` to run on a file, if the file includes out of line
modules, then we reformat those too. So to run on a whole module or crate, you
just need to run on the top file.

You'll probably want to specify the write mode. Currently, there are the
replace, overwrite, display and coverage modes. The replace mode is the default
and overwrites the original files after renaming them. In overwrite mode,
rustfmt does not backup the source files. To print the output to stdout, use the
display mode. The write mode can be set by passing the `--write-mode` flag on
the command line.

`cargo run -- filename --write-mode=display` prints the output of rustfmt to the
screen, for example.


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
