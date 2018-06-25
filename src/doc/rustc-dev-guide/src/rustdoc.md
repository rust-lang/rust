# The walking tour of rustdoc

Rustdoc actually uses the rustc internals directly. It lives in-tree with the
compiler and standard library. This chapter is about how it works.

Rustdoc is implemented entirely within the crate [`librustdoc`][rd]. It runs
the compiler up to the point where we have an internal representation of a
crate (HIR) and the ability to run some queries about the types of items. [HIR]
and [queries] are discussed in the linked chapters.

[HIR]: ./hir.html
[queries]: ./query.html
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

* Use `./x.py build --stage 1 src/libstd src/tools/rustdoc` to make a useable
  rustdoc you can run on other projects.
  * Add `src/libtest` to be able to use `rustdoc --test`.
  * If you've used `rustup toolchain link local /path/to/build/$TARGET/stage1`
    previously, then after the previous build command, `cargo +local doc` will
    Just Work.
* Use `./x.py doc --stage 1 src/libstd` to use this rustdoc to generate the
  standard library docs.
  * The completed docs will be available in `build/$TARGET/doc/std`, though the
    bundle is meant to be used as though you would copy out the `doc` folder to
    a web server, since that's where the CSS/JS and landing page are.
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

## From crate to clean

In `core.rs` are two central items: the `DocContext` struct, and the `run_core`
function. The latter is where rustdoc calls out to rustc to compile a crate to
the point where rustdoc can take over. The former is a state container used
when crawling through a crate to gather its documentation.

The main process of crate crawling is done in `clean/mod.rs` through several
implementations of the `Clean` trait defined within. This is a conversion
trait, which defines one method:

```rust,ignore
pub trait Clean<T> {
    fn clean(&self, cx: &DocContext) -> T;
}
```

`clean/mod.rs` also defines the types for the "cleaned" AST used later on to
render documentation pages. Each usually accompanies an implementation of
`Clean` that takes some AST or HIR type from rustc and converts it into the
appropriate "cleaned" type. "Big" items like modules or associated items may
have some extra processing in its `Clean` implementation, but for the most part
these impls are straightforward conversions. The "entry point" to this module
is the `impl Clean<Crate> for visit_ast::RustdocVisitor`, which is called by
`run_core` above.

You see, I actually lied a little earlier: There's another AST transformation
that happens before the events in `clean/mod.rs`.  In `visit_ast.rs` is the
type `RustdocVisitor`, which *actually* crawls a `hir::Crate` to get the first
intermediate representation, defined in `doctree.rs`. This pass is mainly to
get a few intermediate wrappers around the HIR types and to process visibility
and inlining. This is where `#[doc(inline)]`, `#[doc(no_inline)]`, and
`#[doc(hidden)]` are processed, as well as the logic for whether a `pub use`
should get the full page or a "Reexport" line in the module page.

The other major thing that happens in `clean/mod.rs` is the collection of doc
comments and `#[doc=""]` attributes into a separate field of the Attributes
struct, present on anything that gets hand-written documentation. This makes it
easier to collect this documentation later in the process.

The primary output of this process is a `clean::Crate` with a tree of Items
which describe the publicly-documentable items in the target crate.

### Hot potato

Before moving on to the next major step, a few important "passes" occur over
the documentation.  These do things like combine the separate "attributes" into
a single string and strip leading whitespace to make the document easier on the
markdown parser, or drop items that are not public or deliberately hidden with
`#[doc(hidden)]`. These are all implemented in the `passes/` directory, one
file per pass. By default, all of these passes are run on a crate, but the ones
regarding dropping private/hidden items can be bypassed by passing
`--document-private-items` to rustdoc. Note that unlike the previous set of AST
transformations, the passes happen on the _cleaned_ crate.

(Strictly speaking, you can fine-tune the passes run and even add your own, but
[we're trying to deprecate that][44136]. If you need finer-grain control over
these passes, please let us know!)

[44136]: https://github.com/rust-lang/rust/issues/44136

Here is current (as of this writing) list of passes:

- `propagate-doc-cfg` - propagates `#[doc(cfg(...))]` to child items.
- `collapse-docs` concatenates all document attributes into one document
  attribute. This is necessary because each line of a doc comment is given as a
  separate doc attribute, and this will combine them into a single string with
  line breaks between each attribute.
- `unindent-comments` removes excess indentation on comments in order for
  markdown to like it. This is necessary because the convention for writing
  documentation is to provide a space between the `///` or `//!` marker and the
  text, and stripping that leading space will make the text easier to parse by
  the Markdown parser. (In the past, the markdown parser used was not
  Commonmark- compliant, which caused annoyances with extra whitespace but this
  seems to be less of an issue today.)
- `strip-priv-imports` strips all private import statements (`use`, `extern
  crate`) from a crate. This is necessary because rustdoc will handle *public*
  imports by either inlining the item's documentation to the module or creating
  a "Reexports" section with the import in it. The pass ensures that all of
  these imports are actually relevant to documentation.
- `strip-hidden` and `strip-private` strip all `doc(hidden)` and private items
  from the output. `strip-private` implies `strip-priv-imports`. Basically, the
  goal is to remove items that are not relevant for public documentation.

## From clean to crate

This is where the "second phase" in rustdoc begins. This phase primarily lives
in the `html/` folder, and it all starts with `run()` in `html/render.rs`. This
code is responsible for setting up the `Context`, `SharedContext`, and `Cache`
which are used during rendering, copying out the static files which live in
every rendered set of documentation (things like the fonts, CSS, and JavaScript
that live in `html/static/`), creating the search index, and printing out the
source code rendering, before beginning the process of rendering all the
documentation for the crate.

Several functions implemented directly on `Context` take the `clean::Crate` and
set up some state between rendering items or recursing on a module's child
items. From here the "page rendering" begins, via an enormous `write!()` call
in `html/layout.rs`. The parts that actually generate HTML from the items and
documentation occurs within a series of `std::fmt::Display` implementations and
functions that pass around a `&mut std::fmt::Formatter`. The top-level
implementation that writes out the page body is the `impl<'a> fmt::Display for
Item<'a>` in `html/render.rs`, which switches out to one of several `item_*`
functions based on the kind of `Item` being rendered.

Depending on what kind of rendering code you're looking for, you'll probably
find it either in `html/render.rs` for major items like "what sections should I
print for a struct page" or `html/format.rs` for smaller component pieces like
"how should I print a where clause as part of some other item".

Whenever rustdoc comes across an item that should print hand-written
documentation alongside, it calls out to `html/markdown.rs` which interfaces
with the Markdown parser. This is exposed as a series of types that wrap a
string of Markdown, and implement `fmt::Display` to emit HTML text. It takes
special care to enable certain features like footnotes and tables and add
syntax highlighting to Rust code blocks (via `html/highlight.rs`) before
running the Markdown parser. There's also a function in here
(`find_testable_code`) that specifically scans for Rust code blocks so the
test-runner code can find all the doctests in the crate.

### From soup to nuts

(alternate title: ["An unbroken thread that stretches from those first `Cell`s
to us"][video])

[video]: https://www.youtube.com/watch?v=hOLAGYmUQV0

It's important to note that the AST cleaning can ask the compiler for
information (crucially, `DocContext` contains a `TyCtxt`), but page rendering
cannot. The `clean::Crate` created within `run_core` is passed outside the
compiler context before being handed to `html::render::run`. This means that a
lot of the "supplementary data" that isn't immediately available inside an
item's definition, like which trait is the `Deref` trait used by the language,
needs to be collected during cleaning, stored in the `DocContext`, and passed
along to the `SharedContext` during HTML rendering.  This manifests as a bunch
of shared state, context variables, and `RefCell`s.

Also of note is that some items that come from "asking the compiler" don't go
directly into the `DocContext` - for example, when loading items from a foreign
crate, rustdoc will ask about trait implementations and generate new `Item`s
for the impls based on that information. This goes directly into the returned
`Crate` rather than roundabout through the `DocContext`. This way, these
implementations can be collected alongside the others, right before rendering
the HTML.

## Other tricks up its sleeve

All this describes the process for generating HTML documentation from a Rust
crate, but there are couple other major modes that rustdoc runs in. It can also
be run on a standalone Markdown file, or it can run doctests on Rust code or
standalone Markdown files. For the former, it shortcuts straight to
`html/markdown.rs`, optionally including a mode which inserts a Table of
Contents to the output HTML.

For the latter, rustdoc runs a similar partial-compilation to get relevant
documentation in `test.rs`, but instead of going through the full clean and
render process, it runs a much simpler crate walk to grab *just* the
hand-written documentation. Combined with the aforementioned
"`find_testable_code`" in `html/markdown.rs`, it builds up a collection of
tests to run before handing them off to the libtest test runner. One notable
location in `test.rs` is the function `make_test`, which is where hand-written
doctests get transformed into something that can be executed.

Some extra reading about `make_test` can be found
[here](https://quietmisdreavus.net/code/2018/02/23/how-the-doctests-get-made/).

## Dotting i's and crossing t's

So that's rustdoc's code in a nutshell, but there's more things in the repo
that deal with it. Since we have the full `compiletest` suite at hand, there's
a set of tests in `src/test/rustdoc` that make sure the final HTML is what we
expect in various situations. These tests also use a supplementary script,
`src/etc/htmldocck.py`, that allows it to look through the final HTML using
XPath notation to get a precise look at the output. The full description of all
the commands available to rustdoc tests is in `htmldocck.py`.

In addition, there are separate tests for the search index and rustdoc's
ability to query it. The files in `src/test/rustdoc-js` each contain a
different search query and the expected results, broken out by search tab.
These files are processed by a script in `src/tools/rustdoc-js` and the Node.js
runtime. These tests don't have as thorough of a writeup, but a broad example
that features results in all tabs can be found in `basic.js`. The basic idea is
that you match a given `QUERY` with a set of `EXPECTED` results, complete with
the full item path of each item.
