# Rustdoc internals

<!-- toc -->

This page describes rustdoc's passes and modes. For an overview of rustdoc,
see the ["Rustdoc overview" chapter](./rustdoc.md).

## From crate to clean

In `core.rs` are two central items: the `DocContext` struct, and the
`run_global_ctxt` function. The latter is where rustdoc calls out to rustc to
compile a crate to the point where rustdoc can take over. The former is a state
container used when crawling through a crate to gather its documentation.

The main process of crate crawling is done in `clean/mod.rs` through several
functions with names that start with `clean_`. Each function accepts an `hir`
or `ty` data structure, and outputs a `clean` structure used by rustdoc. For
example, this function for converting lifetimes:

```rust,ignore
fn clean_lifetime<'tcx>(lifetime: &hir::Lifetime, cx: &mut DocContext<'tcx>) -> Lifetime {
    let def = cx.tcx.named_bound_var(lifetime.hir_id);
    if let Some(
        rbv::ResolvedArg::EarlyBound(node_id)
        | rbv::ResolvedArg::LateBound(_, _, node_id)
        | rbv::ResolvedArg::Free(_, node_id),
    ) = def
    {
        if let Some(lt) = cx.args.get(&node_id).and_then(|p| p.as_lt()).cloned() {
            return lt;
        }
    }
    Lifetime(lifetime.ident.name)
}
```

`clean/mod.rs` also defines the types for the "cleaned" AST used later on to
render documentation pages. Each usually accompanies a `clean` function
that takes some AST or HIR type from rustc and converts it into the
appropriate "cleaned" type. "Big" items like modules or associated items may
have some extra processing in its `clean` function, but for the most part
these impls are straightforward conversions. The "entry point" to this module
is `clean::krate`, which is called by
`run_global_ctxt` above.

The first step in `clean::krate` is to invoke `visit_ast::RustdocVisitor` to
process the module tree into an intermediate `visit_ast::Module`. This is the
step that actually crawls the `rustc_hir::Crate`, normalizing various aspects
of name resolution, such as:

  * showing `#[macro_export]`-ed macros at the crate root, regardless of where
    they're defined
  * inlining public `use` exports of private items, or showing a "Reexport"
    line in the module page
  * inlining items with `#[doc(hidden)]` if the base item is hidden but the
    reexport is not
  * handling `#[doc(inline)]` and `#[doc(no_inline)]`
  * handling import globs and cycles, so there are no duplicates or infinite
    directory trees

After this step, `clean::krate` invokes `clean_doc_module`, which actually
converts the HIR items to the cleaned AST. This is also the step where cross-
crate inlining is performed, which requires converting `rustc_middle` data
structures into the cleaned AST instead.

The other major thing that happens in `clean/mod.rs` is the collection of doc
comments and `#[doc=""]` attributes into a separate field of the Attributes
struct, present on anything that gets hand-written documentation. This makes it
easier to collect this documentation later in the process.

The primary output of this process is a `clean::Crate` with a tree of Items
which describe the publicly-documentable items in the target crate.

### Passes anything but a gas station

(alternate title: [hot potato](https://www.youtube.com/watch?v=WNFBIt5HxdY))

Before moving on to the next major step, a few important "passes" occur over
the cleaned AST. Several of these passes are lints and reports, but some of
them mutate or generate new items.

These are all implemented in the `passes/` directory, one file per pass.
By default, all of these passes are run on a crate, but the ones
regarding dropping private/hidden items can be bypassed by passing
`--document-private-items` to rustdoc. Note that unlike the previous set of AST
transformations, the passes are run on the _cleaned_ crate.

Here is the list of passes as of <!-- date-check --> March 2023:

- `calculate-doc-coverage` calculates information used for the `--show-coverage`
  flag.

- `check-doc-test-visibility` runs doctest visibility–related lints. This pass
  runs before `strip-private`, which is why it needs to be separate from
  `run-lints`.

- `collect-intra-doc-links` resolves [intra-doc links](https://doc.rust-lang.org/nightly/rustdoc/write-documentation/linking-to-items-by-name.html).

- `collect-trait-impls` collects trait impls for each item in the crate. For
  example, if we define a struct that implements a trait, this pass will note
  that the struct implements that trait.

- `propagate-doc-cfg` propagates `#[doc(cfg(...))]` to child items.

- `run-lints` runs some of rustdoc's lints, defined in `passes/lint`. This is
  the last pass to run.

  - `bare_urls` detects links that are not linkified, e.g., in Markdown such as
    `Go to https://example.com/.` It suggests wrapping the link with angle brackets:
    `Go to <https://example.com/>.` to linkify it. This is the code behind the <!--
    date-check: may 2022 --> `rustdoc::bare_urls` lint.

  - `check_code_block_syntax` validates syntax inside Rust code blocks
    (<code>```rust</code>)

  - `html_tags` detects invalid HTML (like an unclosed `<span>`)
    in doc comments.

- `strip-hidden` and `strip-private` strip all `doc(hidden)` and private items
  from the output. `strip-private` implies `strip-priv-imports`. Basically, the
  goal is to remove items that are not relevant for public documentation. This
  pass is skipped when `--document-hidden-items` is passed.

- `strip-priv-imports` strips all private import statements (`use`, `extern
  crate`) from a crate. This is necessary because rustdoc will handle *public*
  imports by either inlining the item's documentation to the module or creating
  a "Reexports" section with the import in it. The pass ensures that all of
  these imports are actually relevant to documentation. It is technically
  only run when `--document-private-items` is passed, but `strip-private`
  accomplishes the same thing.

- `strip-private` strips all private items from a crate which cannot be seen
  externally. This pass is skipped when `--document-private-items` is passed.

There is also a `stripper` module in `passes/`, but it is a collection of
utility functions for the `strip-*` passes and is not a pass itself.

## From clean to HTML

This is where the "second phase" in rustdoc begins. This phase primarily lives
in the `formats/` and `html/` folders, and it all starts with
`formats::run_format`. This code is responsible for setting up a type that
`impl FormatRenderer`, which for HTML is [`Context`].

This structure contains methods that get called by `run_format` to drive the
doc rendering, which includes:

* `init` generates `static.files`, as well as search index and `src/`
* `item` generates the item HTML files themselves
* `after_krate` generates other global resources like `all.html`

In `item`, the "page rendering" occurs, via a mixture of [Askama] templates
and manual `write!()` calls, starting in `html/layout.rs`. The parts that have
not been converted to templates occur within a series of `std::fmt::Display`
implementations and functions that pass around a `&mut std::fmt::Formatter`.

The parts that actually generate HTML from the items and documentation start
with `print_item` defined in `html/render/print_item.rs`, which switches out
to one of several `item_*` functions based on kind of `Item` being rendered.

Depending on what kind of rendering code you're looking for, you'll probably
find it either in `html/render/mod.rs` for major items like "what sections
should I print for a struct page" or `html/format/mod.rs` for smaller
component pieces like "how should I print a where clause as part of some other item".

Whenever rustdoc comes across an item that should print hand-written
documentation alongside, it calls out to `html/markdown.rs` which interfaces
with the Markdown parser. This is exposed as a series of types that wrap a
string of Markdown, and implement `fmt::Display` to emit HTML text. It takes
special care to enable certain features like footnotes and tables and add
syntax highlighting to Rust code blocks (via `html/highlight.rs`) before
running the Markdown parser. There's also a function in here
(`find_testable_code`) that specifically scans for Rust code blocks so the
test-runner code can find all the doctests in the crate.

[Askama]: https://docs.rs/askama/latest/askama/

### From soup to nuts

(alternate title: ["An unbroken thread that stretches from those first `Cell`s
to us"][video])

[video]: https://www.youtube.com/watch?v=hOLAGYmUQV0

It's important to note that rustdoc can ask the compiler for type information
directly, even during HTML generation. This [didn't used to be the case], and
a lot of rustdoc's architecture was designed around not doing that, but a
`TyCtxt` is now passed to `formats::renderer::run_format`, which is used to
run generation for both HTML and the
(unstable as of <!-- date-check --> March 2023) JSON format.

[didn't used to be the case]: https://github.com/rust-lang/rust/pull/80090

This change has allowed other changes to remove data from the "clean" AST
that can be easily derived from `TyCtxt` queries, and we'll usually accept
PRs that remove fields from "clean" (it's been soft-deprecated), but this
is complicated from two other constraints that rustdoc runs under:

* Docs can be generated for crates that don't actually pass type checking.
  This is used for generating docs that cover mutually-exclusive platform
  configurations, such as `libstd` having a single package of docs that
  cover all supported operating systems. This means rustdoc has to be able
  to generate docs from HIR.
* Docs can inline across crates. Since crate metadata doesn't contain HIR,
  it must be possible to generate inlined docs from the `rustc_middle` data.

The "clean" AST acts as a common output format for both input formats. There
is also some data in clean that doesn't correspond directly to HIR, such as
synthetic `impl`s for auto traits and blanket `impl`s generated by the
`collect-trait-impls` pass.

Some additional data is stored in
`html::render::context::{Context, SharedContext}`. These two types serve as
ways to segregate rustdoc's data for an eventual future with multithreaded doc
generation, as well as just keeping things organized:

* [`Context`] stores data used for generating the current page, such as its
  path, a list of HTML IDs that have been used (to avoid duplicate `id=""`),
  and the pointer to `SharedContext`.
* [`SharedContext`] stores data that does not vary by page, such as the `tcx`
  pointer, and a list of all types.

[`Context`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc/html/render/context/struct.Context.html
[`SharedContext`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustdoc/html/render/context/struct.SharedContext.html

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
a set of tests in `tests/rustdoc` that make sure the final HTML is what we
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
ability to query it. The files in `tests/rustdoc-js` each contain a
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
$ ./x doc library
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
