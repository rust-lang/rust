# Rust documentations

## Building

To generate all the docs, follow the "Building Documentation" instructions in
the README in the root of the repository. This will convert the distributed
Markdown docs to HTML and generate HTML doc for the books, 'std' and 'extra'
libraries.

To generate HTML documentation from one source file/crate, do something like:

~~~~
rustdoc --output html-doc/ --output-format html ../src/libstd/path.rs
~~~~

(This, of course, requires a working build of the `rustdoc` tool.)

## Additional notes

To generate an HTML version of a doc from Markdown manually, you can do
something like:

~~~~
rustdoc reference.md
~~~~

(`reference.md` being the Rust Reference Manual.)

An overview of how to use the `rustdoc` command is available [in the docs][1].
Further details are available from the command line by with `rustdoc --help`.

[1]: https://github.com/rust-lang/rust/blob/master/src/doc/book/documentation.md
