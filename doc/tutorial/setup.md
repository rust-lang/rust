# Getting started

## Installation

FIXME Fill this in when the installation package is finished.

## Compiling your first program

Rust program files are, by convention, given the extension `.rs`. Say
we have a file `hello.rs` containing this program:

    use std;
    fn main() {
        std::io::println("hello world!");
    }

If the Rust compiler was installed successfully, running `rustc
hello.rs` will produce a binary called `hello` (or `hello.exe`).

If you modify the program to make it invalid (for example, remove the
`use std` line), and then compile it, you'll see an error message like
this:

    hello.rs:2:4: 2:20 error: unresolved modulename: std
    hello.rs:2     std::io::println("hello world!");
                   ^~~~~~~~~~~~~~~~

The Rust compiler tries to provide useful information when it runs
into an error.

## Anatomy of a Rust program

FIXME say something about libs, main, modules, use

## Editing Rust code

There are Vim highlighting and indentation scrips in the Rust source
distribution under `src/etc/vim/`. An Emacs mode can be found at
`[https://github.com/marijnh/rust-mode](https://github.com/marijnh/rust-mode)`.

Other editors are not provided for yet. If you end up writing a Rust
mode for your favorite editor, let us know so that we can link to it.
