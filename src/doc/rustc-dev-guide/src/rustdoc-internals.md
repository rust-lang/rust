# Rustdoc internals

<!-- toc -->

This page describes rustdoc's passes and modes. For an overview of rustdoc,
see [`rustdoc`](./rustdoc.md).

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
type `RustdocVisitor`, which *actually* crawls a `rustc_hir::Crate` to get the first
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
the documentation. These do things like combine the separate "attributes" into
a single string and strip leading whitespace to make the document easier on the
markdown parser, or drop items that are not public or deliberately hidden with
`#[doc(hidden)]`. These are all implemented in the `passes/` directory, one
file per pass. By default, all of these passes are run on a crate, but the ones
regarding dropping private/hidden items can be bypassed by passing
`--document-private-items` to rustdoc. Note that unlike the previous set of AST
transformations, the passes are run on the _cleaned_ crate.

(Strictly speaking, you can fine-tune the passes run and even add your own, but
[we're trying to deprecate that][44136]. If you need finer-grain control over
these passes, please let us know!)

[44136]: https://github.com/rust-lang/rust/issues/44136

Here is the list of passes as of February 2021 <!-- date: 2021-02 -->:

- `calculate-doc-coverage` calculates information used for the `--show-coverage`
  flag.

- `check-code-block-syntax` validates syntax inside Rust code blocks
  (`` ```rust ``)

- `check-invalid-html-tags` detects invalid HTML (like an unclosed `<span>`)
  in doc comments.

- `check-non-autolinks` detects links that could or should be written using
  angle brackets (the code behind the nightly-only <!-- date: 2021-02 -->
  `non_autolinks` lint).

- `collapse-docs` concatenates all document attributes into one document
  attribute. This is necessary because each line of a doc comment is given as a
  separate doc attribute, and this will combine them into a single string with
  line breaks between each attribute.

- `collect-intra-doc-links` resolves [intra-doc links](https://doc.rust-lang.org/rustdoc/linking-to-items-by-name.html).

- `collect-trait-impls` collects trait impls for each item in the crate. For
  example, if we define a struct that implements a trait, this pass will note
  that the struct implements that trait.

- `doc-test-lints` runs various lints on the doctests.

- `propagate-doc-cfg` propagates `#[doc(cfg(...))]` to child items.

- `strip-priv-imports` strips all private import statements (`use`, `extern
  crate`) from a crate. This is necessary because rustdoc will handle *public*
  imports by either inlining the item's documentation to the module or creating
  a "Reexports" section with the import in it. The pass ensures that all of
  these imports are actually relevant to documentation.

- `strip-hidden` and `strip-private` strip all `doc(hidden)` and private items
  from the output. `strip-private` implies `strip-priv-imports`. Basically, the
  goal is to remove items that are not relevant for public documentation.

- `unindent-comments` removes excess indentation on comments in order for the
  Markdown to be parsed correctly. This is necessary because the convention for
  writing documentation is to provide a space between the `///` or `//!` marker
  and the doc text, but Markdown is whitespace-sensitive. For example, a block
  of text with four-space indentation is parsed as a code block, so if we didn't
  unindent comments, these list items

  ```rust,ignore
  /// A list:
  ///
  ///    - Foo
  ///    - Bar
  ```

  would be parsed as if they were in a code block, which is likely not what the
  user intended.

There is also a `stripper` module in `passes/`, but it is a collection of
utility functions for the `strip-*` passes and is not a pass itself.

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
tests to run before handing them off to the test runner. One notable
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
the commands available to rustdoc tests (e.g. [`@has`] and [`@matches`]) is in
[`htmldocck.py`].

To use multiple crates in a rustdoc test, add `// aux-build:filename.rs`
to the top of the test file. `filename.rs` should be placed in an `auxiliary`
directory relative to the test file with the comment. If you need to build
docs for the auxiliary file, use `// build-aux-docs`.

In addition, there are separate tests for the search index and rustdoc's
ability to query it. The files in `src/test/rustdoc-js` each contain a
different search query and the expected results, broken out by search tab.
These files are processed by a script in `src/tools/rustdoc-js` and the Node.js
runtime. These tests don't have as thorough of a writeup, but a broad example
that features results in all tabs can be found in `basic.js`. The basic idea is
that you match a given `QUERY` with a set of `EXPECTED` results, complete with
the full item path of each item.

[`htmldocck.py`]: https://github.com/rust-lang/rust/blob/master/src/etc/htmldocck.py
[`@has`]: https://github.com/rust-lang/rust/blob/master/src/etc/htmldocck.py#L39
[`@matches`]: https://github.com/rust-lang/rust/blob/master/src/etc/htmldocck.py#L44

## Testing locally

Some features of the generated HTML documentation might require local
storage to be used across pages, which doesn't work well without an HTTP
server. To test these features locally, you can run a local HTTP server, like
this:

```bash
$ ./x.py doc library/std --stage 1
# The documentation has been generated into `build/[YOUR ARCH]/doc`.
$ python3 -m http.server -d build/[YOUR ARCH]/doc
```

Now you can browse your documentation just like you would if it was hosted
on the internet. For example, the url for `std` will be `/std/".

## See also

- The [`rustdoc` api docs]
- [An overview of `rustdoc`](./rustdoc.md)
- [The rustdoc user guide]

[`rustdoc` api docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc/
[The rustdoc user guide]: https://doc.rust-lang.org/nightly/rustdoc/
