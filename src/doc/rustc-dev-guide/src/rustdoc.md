# Rustdoc overview

`rustdoc` lives in-tree with the
compiler and standard library. This chapter is about how it works.
For information about Rustdoc's features and how to use them, see
the [Rustdoc book](https://doc.rust-lang.org/nightly/rustdoc/).
For more details about how rustdoc works, see the
["Rustdoc internals" chapter][Rustdoc internals].

[Rustdoc internals]: ./rustdoc-internals.md

<!-- toc -->

`rustdoc` uses `rustc` internals (and, of course, the standard library), so you
will have to build the compiler and `std` once before you can build `rustdoc`.

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

* Run `./x setup tools` before getting started. This will configure `x`
  with nice settings for developing rustdoc and other tools, including
  downloading a copy of rustc rather than building it.
* Use `./x check rustdoc` to quickly check for compile errors.
* Use `./x build library rustdoc` to make a usable
  rustdoc you can run on other projects.
  * Add `library/test` to be able to use `rustdoc --test`.
  * Run `rustup toolchain link stage2 build/host/stage2` to add a
    custom toolchain called `stage2` to your rustup environment. After
    running that, `cargo +stage2 doc` in any directory will build with
    your locally-compiled rustdoc.
* Use `./x doc library` to use this rustdoc to generate the
  standard library docs.
  * The completed docs will be available in `build/host/doc` (under `core`, `alloc`, and `std`).
  * If you want to copy those docs to a webserver, copy all of
    `build/host/doc`, since that's where the CSS, JS, fonts, and landing
    page are.
  * For frontend debugging, disable the `rust.docs-minification` option in [`bootstrap.toml`].
* Use `./x test tests/rustdoc*` to run the tests using a stage1
  rustdoc.
  * See [Rustdoc internals] for more information about tests.

[`bootstrap.toml`]: ./building/how-to-build-and-run.md

## Code structure

All paths in this section are relative to `src/librustdoc/` in the rust-lang/rust repository.

* Most of the HTML printing code is in `html/format.rs` and `html/render/mod.rs`.
  It's in a bunch of functions returning `impl std::fmt::Display`.
* The data types that get rendered by the functions mentioned above are defined in `clean/types.rs`.
  The functions responsible for creating them from the `HIR` and the `rustc_middle::ty` IR
  live in `clean/mod.rs`.
* The bits specific to using rustdoc as a test harness are in
  `doctest.rs`.
* The Markdown renderer is loaded up in `html/markdown.rs`, including functions
  for extracting doctests from a given block of Markdown.
* Frontend CSS and JavaScript are stored in `html/static/`.
  * Re. JavaScript, type annotations are written using [TypeScript-flavored JSDoc]
comments and an external `.d.ts` file.
    This way, the code itself remains plain, valid JavaScript.
    We only use `tsc` as a linter.

[TypeScript-flavored JSDoc]: https://www.typescriptlang.org/docs/handbook/jsdoc-supported-types.html

## Tests

`rustdoc`'s integration tests are split across several test suites.
See [Rustdoc tests suites](tests/compiletest.md#rustdoc-test-suites) for more details.

## Constraints

We try to make rustdoc work reasonably well with JavaScript disabled, and when
browsing local files. We support
[these browsers](https://rust-lang.github.io/rfcs/1985-tiered-browser-support.html#supported-browsers).

Supporting local files (`file:///` URLs) brings some surprising restrictions.
Certain browser features that require secure origins, like `localStorage` and
Service Workers, don't work reliably. We can still use such features but we
should make sure pages are still usable without them.

Rustdoc [does not type-check function bodies][platform-specific docs].
This works by [overriding the built-in queries for typeck][override queries],
by [silencing name resolution errors], and by [not resolving opaque types].
This comes with several caveats: in particular, rustdoc *cannot* run any parts of the compiler that
require type-checking bodies; for example it cannot generate `.rlib` files or run most lints.
We want to move away from this model eventually, but we need some alternative for
[the people using it][async-std]; see [various][zulip stop accepting broken code]
[previous][rustdoc meeting 2024-07-08] [zulip][compiler meeting 2023-01-26] [discussion][notriddle rfc].
For examples of code that breaks if this hack is removed, see
[`tests/rustdoc-ui/error-in-impl-trait`].

[platform-specific docs]: https://doc.rust-lang.org/rustdoc/advanced-features.html#interactions-between-platform-specific-docs
[override queries]: https://github.com/rust-lang/rust/blob/52bf0cf795dfecc8b929ebb1c1e2545c3f41d4c9/src/librustdoc/core.rs#L299-L323
[silencing name resolution errors]: https://github.com/rust-lang/rust/blob/52bf0cf795dfecc8b929ebb1c1e2545c3f41d4c9/compiler/rustc_resolve/src/late.rs#L4517
[not resolving opaque types]: https://github.com/rust-lang/rust/blob/52bf0cf795dfecc8b929ebb1c1e2545c3f41d4c9/compiler/rustc_hir_analysis/src/check/check.rs#L188-L194
[async-std]: https://github.com/rust-lang/rust/issues/75100
[rustdoc meeting 2024-07-08]: https://rust-lang.zulipchat.com/#narrow/channel/393423-t-rustdoc.2Fmeetings/topic/meeting.202024-07-08/near/449969836
[compiler meeting 2023-01-26]: https://rust-lang.zulipchat.com/#narrow/channel/238009-t-compiler.2Fmeetings/topic/.5Bweekly.5D.202023-01-26/near/323755789
[zulip stop accepting broken code]: https://rust-lang.zulipchat.com/#narrow/stream/266220-rustdoc/topic/stop.20accepting.20broken.20code
[notriddle rfc]: https://rust-lang.zulipchat.com/#narrow/channel/266220-t-rustdoc/topic/Pre-RFC.3A.20stop.20accepting.20broken.20code
[`tests/rustdoc-ui/error-in-impl-trait`]: https://github.com/rust-lang/rust/tree/163cb4ea3f0ae3bc7921cc259a08a7bf92e73ee6/tests/rustdoc-ui/error-in-impl-trait

## Multiple runs, same output directory

Rustdoc can be run multiple times for varying inputs, with its output set to the
same directory. That's how cargo produces documentation for dependencies of the
current crate. It can also be done manually if a user wants a big
documentation bundle with all of the docs they care about.

HTML is generated independently for each crate, but there is some cross-crate
information that we update as we add crates to the output directory:

 - `crates<SUFFIX>.js` holds a list of all crates in the output directory.
 - `search-index<SUFFIX>.js` holds a list of all searchable items.
 - For each trait, there is a file under `implementors/.../trait.TraitName.js`
   containing a list of implementors of that trait. The implementors may be in
   different crates than the trait, and the JS file is updated as we discover
   new ones.

## Use cases

There are a few major use cases for rustdoc that you should keep in mind when
working on it:

### Standard library docs

These are published at <https://doc.rust-lang.org/std> as part of the Rust release
process. Stable releases are also uploaded to specific versioned URLs like
<https://doc.rust-lang.org/1.57.0/std/>. Beta and nightly docs are published to
<https://doc.rust-lang.org/beta/std/> and <https://doc.rust-lang.org/nightly/std/>.
The docs are uploaded with the [promote-release
tool](https://github.com/rust-lang/promote-release) and served from S3 with
CloudFront.

The standard library docs contain five crates: alloc, core, proc_macro, std, and
test.

### docs.rs

When crates are published to crates.io, docs.rs automatically builds
and publishes their documentation, for instance at
<https://docs.rs/serde/latest/serde/>. It always builds with the current nightly
rustdoc, so any changes you land in rustdoc are "insta-stable" in that they will
have an immediate public effect on docs.rs. Old documentation is not rebuilt, so
you will see some variation in UI when browsing old releases in docs.rs. Crate
authors can request rebuilds, which will be run with the latest rustdoc.

Docs.rs performs some transformations on rustdoc's output in order to save
storage and display a navigation bar at the top. In particular, certain static
files, like main.js and rustdoc.css, may be shared across multiple invocations
of the same version of rustdoc. Others, like crates.js and sidebar-items.js, are
different for different invocations. Still others, like fonts, will never
change. These categories are distinguished using the `SharedResource` enum in
`src/librustdoc/html/render/write_shared.rs`

Documentation on docs.rs is always generated for a single crate at a time, so
the search and sidebar functionality don't include dependencies of the current
crate.

### Locally generated docs

Crate authors can run `cargo doc --open` in crates they have checked
out locally to see the docs. This is useful to check that the docs they
are writing are useful and display correctly. It can also be useful for
people to view documentation on crates they aren't authors of, but want to
use. In both cases, people may use `--document-private-items` Cargo flag to
see private methods, fields, and so on, which are normally not displayed.

By default `cargo doc` will generate documentation for a crate and all of its
dependencies. That can result in a very large documentation bundle, with a large
(and slow) search corpus. The Cargo flag `--no-deps` inhibits that behavior and
generates docs for just the crate.

### Self-hosted project docs

Some projects like to host their own documentation. For example:
<https://docs.serde.rs/>. This is easy to do by locally generating docs, and
simply copying them to a web server. Rustdoc's HTML output can be extensively
customized by flags. Users can add a theme, set the default theme, and inject
arbitrary HTML. See `rustdoc --help` for details.
