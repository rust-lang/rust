# Building documentation

This chapter describes how to build documentation of toolchain components,
either in whole or individually.

- Document everything

  This uses `rustdoc` from the beta toolchain,
  so will produce (slightly) different output to stage 1 rustdoc,
  as `rustdoc` is under active development:

  ```bash
  ./x.py doc
  ```

  If you want to be sure the documentation looks the same as on CI:

  ```bash
  ./x.py doc --stage 1
  ```

  First,
  the compiler and rustdoc get built to make sure everything is okay,
  and then it documents the files.

- Much like running individual tests or building specific components,
  you can build just the documentation you want:

  ```bash
  ./x.py doc src/doc/book
  ./x.py doc src/doc/nomicon
  ./x.py doc compiler library
  ```

- Document internal rustc items

  Compiler documentation is not built by default.
  To enable it, modify `config.toml`:

  ```toml
  [build]
  compiler-docs = true
  ```

  Note that when enabled,
  documentation for internal compiler items will also be built.

  NOTE: The documentation for the compiler is found at [this link].

[this link]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/
