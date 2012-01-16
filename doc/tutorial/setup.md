# Getting started

## Installation

FIXME Fill this in when the installation package is finished.

## Compiling your first program

Rust program files are, by convention, given the extension `.rs`. Say
we have a file `hello.rs` containing this program:

    use std;
    fn main(args: [str]) {
        std::io::println("hello world from '" + args[0] + "'!");
    }

If the Rust compiler was installed successfully, running `rustc
hello.rs` will produce a binary called `hello` (or `hello.exe`).

If you modify the program to make it invalid (for example, remove the
`use std` line), and then compile it, you'll see an error message like
this:

    ## notrust
    hello.rs:2:4: 2:20 error: unresolved modulename: std
    hello.rs:2     std::io::println("hello world!");
                   ^~~~~~~~~~~~~~~~

The Rust compiler tries to provide useful information when it runs
into an error.

## Anatomy of a Rust program

In its simplest form, a Rust program is simply a `.rs` file with some
types and functions defined in it. If it has a `main` function, it can
be compiled to an executable. Rust does not allow code that's not a
declaration to appear at the top level of the file—all statements must
live inside a function.

Rust programs can also be compiled as libraries, and included in other
programs. The `use std` directive that appears at the top of a lot of
examples imports the [standard library][std]. This is described in more
detail [later on](mod.html).

[std]: http://doc.rust-lang.org/doc/std/index/General.html

## Editing Rust code

There are Vim highlighting and indentation scrips in the Rust source
distribution under `src/etc/vim/`, and an emacs mode under
`src/etc/emacs/`.

[rust-mode]: https://github.com/marijnh/rust-mode

Other editors are not provided for yet. If you end up writing a Rust
mode for your favorite editor, let us know so that we can link to it.
