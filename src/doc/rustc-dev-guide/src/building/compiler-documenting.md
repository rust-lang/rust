# Building documentation

This chapter describes how to build documentation of toolchain components,
like the standard library (std) or the compiler (rustc).

- Document everything

  This uses `rustdoc` from the beta toolchain,
  so will produce (slightly) different output to stage 1 rustdoc,
  as rustdoc is under active development:

  ```bash
  ./x doc
  ```

  If you want to be sure the documentation looks the same as on CI:

  ```bash
  ./x doc --stage 1
  ```

  This ensures that (current) rustdoc gets built,
  then that is used to document the components.

- Much like running individual tests or building specific components,
  you can build just the documentation you want:

  ```bash
  ./x doc src/doc/book
  ./x doc src/doc/nomicon
  ./x doc compiler library
  ```

  See [the nightly docs index page](https://doc.rust-lang.org/nightly/) for a full list of books.

- Document internal rustc items

  Compiler documentation is not built by default.
  To create it by default with `x doc`, modify `bootstrap.toml`:

  ```toml
  [build]
  compiler-docs = true
  ```

  Note that when enabled,
  documentation for internal compiler items will also be built.

  NOTE: The documentation for the compiler is found at [this link].

[this link]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/
