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
  build.compiler-docs = true
  ```

  Note that when enabled,
  documentation for internal compiler items will also be built.

  NOTE: The documentation for the compiler is found at [this link].

[this link]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/

## Contributing documentation

Documentation improvements are very welcome.
The source of `doc.rust-lang.org`
is located in [`src/doc`] in the tree, and standard API documentation is generated
from the source code itself (e.g. [`library/std/src/lib.rs`][std-root]). Documentation pull requests
function in the same way as other pull requests.

[`src/doc`]: https://github.com/rust-lang/rust/tree/HEAD/src/doc
[std-root]: https://github.com/rust-lang/rust/blob/HEAD/library/std/src/lib.rs#L1

To find documentation-related issues, use the [A-docs label].

You can find documentation style guidelines in [RFC 1574].

To build the standard library documentation, use `x doc --stage 1 library --open`.
To build the documentation for a book (e.g. the unstable book), use `x doc src/doc/unstable-book`.
Results should appear in `build/host/doc`, as well as automatically open in your default browser.
See [Building Documentation](#building-documentation) for more
information.

You can also use `rustdoc` directly to check small fixes.
For example, `rustdoc src/doc/reference.md` will render reference to `doc/reference.html`.
The CSS might be messed up, but you can verify that the HTML is right.

Please notice that we don't accept typography/spellcheck fixes to **internal documentation**
as it's usually not worth the churn or the review time.
Examples of internal documentation are code comments and rustc API docs.
However, feel free to fix those if accompanied by other improvements in the same PR.

[A-docs label]: https://github.com/rust-lang/rust/issues?q=is%3Aopen%20is%3Aissue%20label%3AA-docs
[RFC 1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md#appendix-a-full-conventions-text
