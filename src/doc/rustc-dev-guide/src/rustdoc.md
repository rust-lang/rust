# Rustdoc overview

Rustdoc actually uses the rustc internals directly. It lives in-tree with the
compiler and standard library. This chapter is about how it works.
For information about Rustdoc's features and how to use them, see
the [Rustdoc book](https://doc.rust-lang.org/nightly/rustdoc/).
For more details about how rustdoc works, see the
["Rustdoc internals" chapter][Rustdoc internals].

[Rustdoc internals]: ./rustdoc-internals.md

Rustdoc is implemented entirely within the crate [`librustdoc`][rd]. It runs
the compiler up to the point where we have an internal representation of a
crate (HIR) and the ability to run some queries about the types of items. [HIR]
and [queries] are discussed in the linked chapters.

[HIR]: ./hir.md
[queries]: ./query.md
[rd]: https://github.com/rust-lang/rust/tree/master/src/librustdoc

`librustdoc` performs two major steps after that to render a set of
documentation:

* "Clean" the AST into a form that's more suited to creating documentation (and
  slightly more resistant to churn in the compiler).
* Use this cleaned AST to render a crate's documentation, one page at a time.

Naturally, there's more than just this, and those descriptions simplify out
lots of details, but that's the high-level overview.

(Side note: `librustdoc` is a library crate! The `rustdoc` binary is created
using the project in [`src/tools/rustdoc`][bin]. Note that literally all that
does is call the `main()` that's in this crate's `lib.rs`, though.)

[bin]: https://github.com/rust-lang/rust/tree/master/src/tools/rustdoc

## Cheat sheet

* Use `./x.py build` to make a usable
  rustdoc you can run on other projects.
  * Add `library/test` to be able to use `rustdoc --test`.
  * If you've used `rustup toolchain link local /path/to/build/$TARGET/stage1`
    previously, then after the previous build command, `cargo +local doc` will
    Just Work.
* Use `./x.py doc --stage 1 library/std` to use this rustdoc to generate the
  standard library docs.
  * The completed docs will be available in `build/$TARGET/doc/std`, though the
    bundle is meant to be used as though you would copy out the `doc` folder to
    a web server, since that's where the CSS/JS and landing page are.
* Use `x.py test src/test/rustdoc*` to run the tests using a stage1 rustdoc.
  * See [Rustdoc internals] for more information about tests.
* Most of the HTML printing code is in `html/format.rs` and `html/render.rs`.
  It's in a bunch of `fmt::Display` implementations and supplementary
  functions.
* The types that got `Display` impls above are defined in `clean/mod.rs`, right
  next to the custom `Clean` trait used to process them out of the rustc HIR.
* The bits specific to using rustdoc as a test harness are in `test.rs`.
* The Markdown renderer is loaded up in `html/markdown.rs`, including functions
  for extracting doctests from a given block of Markdown.
* The tests on rustdoc *output* are located in `src/test/rustdoc`, where
  they're handled by the test runner of rustbuild and the supplementary script
  `src/etc/htmldocck.py`.
* Tests on search index generation are located in `src/test/rustdoc-js`, as a
  series of JavaScript files that encode queries on the standard library search
  index and expected results.
