#![allow(clippy::lint_without_lint_pass)]

use clippy_config::Conf;
use clippy_utils::attrs::is_doc_hidden;
use clippy_utils::diagnostics::{span_lint, span_lint_and_help, span_lint_and_then};
use clippy_utils::source::snippet_opt;
use clippy_utils::{is_entrypoint_fn, is_trait_impl_item};
use pulldown_cmark::Event::{
    Code, DisplayMath, End, FootnoteReference, HardBreak, Html, InlineHtml, InlineMath, Rule, SoftBreak, Start,
    TaskListMarker, Text,
};
use pulldown_cmark::Tag::{BlockQuote, CodeBlock, FootnoteDefinition, Heading, Item, Link, Paragraph};
use pulldown_cmark::{BrokenLink, CodeBlockKind, CowStr, Options, TagEnd};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::{Attribute, ImplItemKind, ItemKind, Node, Safety, TraitItemKind};
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_resolve::rustdoc::{
    DocFragment, add_doc_fragment, attrs_to_doc_fragments, main_body_opts, source_span_for_markdown_range,
    span_of_fragments,
};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::edition::Edition;
use std::ops::Range;
use url::Url;

mod doc_comment_double_space_linebreaks;
mod include_in_doc_without_cfg;
mod lazy_continuation;
mod link_with_quotes;
mod markdown;
mod missing_headers;
mod needless_doctest_main;
mod suspicious_doc_comments;
mod too_long_first_doc_paragraph;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the presence of `_`, `::` or camel-case words
    /// outside ticks in documentation.
    ///
    /// ### Why is this bad?
    /// *Rustdoc* supports markdown formatting, `_`, `::` and
    /// camel-case probably indicates some code which should be included between
    /// ticks. `_` can also be used for emphasis in markdown, this lint tries to
    /// consider that.
    ///
    /// ### Known problems
    /// Lots of bad docs won’t be fixed, what the lint checks
    /// for is limited, and there are still false positives. HTML elements and their
    /// content are not linted.
    ///
    /// In addition, when writing documentation comments, including `[]` brackets
    /// inside a link text would trip the parser. Therefore, documenting link with
    /// `[`SmallVec<[T; INLINE_CAPACITY]>`]` and then [`SmallVec<[T; INLINE_CAPACITY]>`]: SmallVec
    /// would fail.
    ///
    /// ### Examples
    /// ```no_run
    /// /// Do something with the foo_bar parameter. See also
    /// /// that::other::module::foo.
    /// // ^ `foo_bar` and `that::other::module::foo` should be ticked.
    /// fn doit(foo_bar: usize) {}
    /// ```
    ///
    /// ```no_run
    /// // Link text with `[]` brackets should be written as following:
    /// /// Consume the array and return the inner
    /// /// [`SmallVec<[T; INLINE_CAPACITY]>`][SmallVec].
    /// /// [SmallVec]: SmallVec
    /// fn main() {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DOC_MARKDOWN,
    pedantic,
    "presence of `_`, `::` or camel-case outside backticks in documentation"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for links with code directly adjacent to code text:
    /// `` [`MyItem`]`<`[`u32`]`>` ``.
    ///
    /// ### Why is this bad?
    /// It can be written more simply using HTML-style `<code>` tags.
    ///
    /// ### Example
    /// ```no_run
    /// //! [`first`](x)`second`
    /// ```
    /// Use instead:
    /// ```no_run
    /// //! <code>[first](x)second</code>
    /// ```
    #[clippy::version = "1.86.0"]
    pub DOC_LINK_CODE,
    nursery,
    "link with code back-to-back with other code"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the doc comments of publicly visible
    /// unsafe functions and warns if there is no `# Safety` section.
    ///
    /// ### Why is this bad?
    /// Unsafe functions should document their safety
    /// preconditions, so that users can be sure they are using them safely.
    ///
    /// ### Examples
    /// ```no_run
    ///# type Universe = ();
    /// /// This function should really be documented
    /// pub unsafe fn start_apocalypse(u: &mut Universe) {
    ///     unimplemented!();
    /// }
    /// ```
    ///
    /// At least write a line about safety:
    ///
    /// ```no_run
    ///# type Universe = ();
    /// /// # Safety
    /// ///
    /// /// This function should not be called before the horsemen are ready.
    /// pub unsafe fn start_apocalypse(u: &mut Universe) {
    ///     unimplemented!();
    /// }
    /// ```
    #[clippy::version = "1.39.0"]
    pub MISSING_SAFETY_DOC,
    style,
    "`pub unsafe fn` without `# Safety` docs"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks the doc comments of publicly visible functions that
    /// return a `Result` type and warns if there is no `# Errors` section.
    ///
    /// ### Why is this bad?
    /// Documenting the type of errors that can be returned from a
    /// function can help callers write code to handle the errors appropriately.
    ///
    /// ### Examples
    /// Since the following function returns a `Result` it has an `# Errors` section in
    /// its doc comment:
    ///
    /// ```no_run
    ///# use std::io;
    /// /// # Errors
    /// ///
    /// /// Will return `Err` if `filename` does not exist or the user does not have
    /// /// permission to read it.
    /// pub fn read(filename: String) -> io::Result<String> {
    ///     unimplemented!();
    /// }
    /// ```
    #[clippy::version = "1.41.0"]
    pub MISSING_ERRORS_DOC,
    pedantic,
    "`pub fn` returns `Result` without `# Errors` in doc comment"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks the doc comments of publicly visible functions that
    /// may panic and warns if there is no `# Panics` section.
    ///
    /// ### Why is this bad?
    /// Documenting the scenarios in which panicking occurs
    /// can help callers who do not want to panic to avoid those situations.
    ///
    /// ### Examples
    /// Since the following function may panic it has a `# Panics` section in
    /// its doc comment:
    ///
    /// ```no_run
    /// /// # Panics
    /// ///
    /// /// Will panic if y is 0
    /// pub fn divide_by(x: i32, y: i32) -> i32 {
    ///     if y == 0 {
    ///         panic!("Cannot divide by 0")
    ///     } else {
    ///         x / y
    ///     }
    /// }
    /// ```
    ///
    /// Individual panics within a function can be ignored with `#[expect]` or
    /// `#[allow]`:
    ///
    /// ```no_run
    /// # use std::num::NonZeroUsize;
    /// pub fn will_not_panic(x: usize) {
    ///     #[expect(clippy::missing_panics_doc, reason = "infallible")]
    ///     let y = NonZeroUsize::new(1).unwrap();
    ///
    ///     // If any panics are added in the future the lint will still catch them
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub MISSING_PANICS_DOC,
    pedantic,
    "`pub fn` may panic without `# Panics` in doc comment"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `fn main() { .. }` in doctests
    ///
    /// ### Why is this bad?
    /// The test can be shorter (and likely more readable)
    /// if the `fn main()` is left implicit.
    ///
    /// ### Examples
    /// ```no_run
    /// /// An example of a doctest with a `main()` function
    /// ///
    /// /// # Examples
    /// ///
    /// /// ```
    /// /// fn main() {
    /// ///     // this needs not be in an `fn`
    /// /// }
    /// /// ```
    /// fn needless_main() {
    ///     unimplemented!();
    /// }
    /// ```
    #[clippy::version = "1.40.0"]
    pub NEEDLESS_DOCTEST_MAIN,
    style,
    "presence of `fn main() {` in code examples"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[test]` in doctests unless they are marked with
    /// either `ignore`, `no_run` or `compile_fail`.
    ///
    /// ### Why is this bad?
    /// Code in examples marked as `#[test]` will somewhat
    /// surprisingly not be run by `cargo test`. If you really want
    /// to show how to test stuff in an example, mark it `no_run` to
    /// make the intent clear.
    ///
    /// ### Examples
    /// ```no_run
    /// /// An example of a doctest with a `main()` function
    /// ///
    /// /// # Examples
    /// ///
    /// /// ```
    /// /// #[test]
    /// /// fn equality_works() {
    /// ///     assert_eq!(1_u8, 1);
    /// /// }
    /// /// ```
    /// fn test_attr_in_doctest() {
    ///     unimplemented!();
    /// }
    /// ```
    #[clippy::version = "1.76.0"]
    pub TEST_ATTR_IN_DOCTEST,
    suspicious,
    "presence of `#[test]` in code examples"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects the syntax `['foo']` in documentation comments (notice quotes instead of backticks)
    /// outside of code blocks
    /// ### Why is this bad?
    /// It is likely a typo when defining an intra-doc link
    ///
    /// ### Example
    /// ```no_run
    /// /// See also: ['foo']
    /// fn bar() {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// /// See also: [`foo`]
    /// fn bar() {}
    /// ```
    #[clippy::version = "1.63.0"]
    pub DOC_LINK_WITH_QUOTES,
    pedantic,
    "possible typo for an intra-doc link"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the doc comments of publicly visible
    /// safe functions and traits and warns if there is a `# Safety` section.
    ///
    /// ### Why restrict this?
    /// Safe functions and traits are safe to implement and therefore do not
    /// need to describe safety preconditions that users are required to uphold.
    ///
    /// ### Examples
    /// ```no_run
    ///# type Universe = ();
    /// /// # Safety
    /// ///
    /// /// This function should not be called before the horsemen are ready.
    /// pub fn start_apocalypse_but_safely(u: &mut Universe) {
    ///     unimplemented!();
    /// }
    /// ```
    ///
    /// The function is safe, so there shouldn't be any preconditions
    /// that have to be explained for safety reasons.
    ///
    /// ```no_run
    ///# type Universe = ();
    /// /// This function should really be documented
    /// pub fn start_apocalypse(u: &mut Universe) {
    ///     unimplemented!();
    /// }
    /// ```
    #[clippy::version = "1.67.0"]
    pub UNNECESSARY_SAFETY_DOC,
    restriction,
    "`pub fn` or `pub trait` with `# Safety` docs"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects the use of outer doc comments (`///`, `/**`) followed by a bang (`!`): `///!`
    ///
    /// ### Why is this bad?
    /// Triple-slash comments (known as "outer doc comments") apply to items that follow it.
    /// An outer doc comment followed by a bang (i.e. `///!`) has no specific meaning.
    ///
    /// The user most likely meant to write an inner doc comment (`//!`, `/*!`), which
    /// applies to the parent item (i.e. the item that the comment is contained in,
    /// usually a module or crate).
    ///
    /// ### Known problems
    /// Inner doc comments can only appear before items, so there are certain cases where the suggestion
    /// made by this lint is not valid code. For example:
    /// ```rust
    /// fn foo() {}
    /// ///!
    /// fn bar() {}
    /// ```
    /// This lint detects the doc comment and suggests changing it to `//!`, but an inner doc comment
    /// is not valid at that position.
    ///
    /// ### Example
    /// In this example, the doc comment is attached to the *function*, rather than the *module*.
    /// ```no_run
    /// pub mod util {
    ///     ///! This module contains utility functions.
    ///
    ///     pub fn dummy() {}
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// pub mod util {
    ///     //! This module contains utility functions.
    ///
    ///     pub fn dummy() {}
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub SUSPICIOUS_DOC_COMMENTS,
    suspicious,
    "suspicious usage of (outer) doc comments"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects documentation that is empty.
    /// ### Why is this bad?
    /// Empty docs clutter code without adding value, reducing readability and maintainability.
    /// ### Example
    /// ```no_run
    /// ///
    /// fn returns_true() -> bool {
    ///     true
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn returns_true() -> bool {
    ///     true
    /// }
    /// ```
    #[clippy::version = "1.78.0"]
    pub EMPTY_DOCS,
    suspicious,
    "docstrings exist but documentation is empty"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// In CommonMark Markdown, the language used to write doc comments, a
    /// paragraph nested within a list or block quote does not need any line
    /// after the first one to be indented or marked. The specification calls
    /// this a "lazy paragraph continuation."
    ///
    /// ### Why is this bad?
    ///
    /// This is easy to write but hard to read. Lazy continuations makes
    /// unintended markers hard to see, and make it harder to deduce the
    /// document's intended structure.
    ///
    /// ### Example
    ///
    /// This table is probably intended to have two rows,
    /// but it does not. It has zero rows, and is followed by
    /// a block quote.
    /// ```no_run
    /// /// Range | Description
    /// /// ----- | -----------
    /// /// >= 1  | fully opaque
    /// /// < 1   | partially see-through
    /// fn set_opacity(opacity: f32) {}
    /// ```
    ///
    /// Fix it by escaping the marker:
    /// ```no_run
    /// /// Range | Description
    /// /// ----- | -----------
    /// /// \>= 1 | fully opaque
    /// /// < 1   | partially see-through
    /// fn set_opacity(opacity: f32) {}
    /// ```
    ///
    /// This example is actually intended to be a list:
    /// ```no_run
    /// /// * Do nothing.
    /// /// * Then do something. Whatever it is needs done,
    /// /// it should be done right now.
    /// # fn do_stuff() {}
    /// ```
    ///
    /// Fix it by indenting the list contents:
    /// ```no_run
    /// /// * Do nothing.
    /// /// * Then do something. Whatever it is needs done,
    /// ///   it should be done right now.
    /// # fn do_stuff() {}
    /// ```
    #[clippy::version = "1.80.0"]
    pub DOC_LAZY_CONTINUATION,
    style,
    "require every line of a paragraph to be indented and marked"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Detects overindented list items in doc comments where the continuation
    /// lines are indented more than necessary.
    ///
    /// ### Why is this bad?
    ///
    /// Overindented list items in doc comments can lead to inconsistent and
    /// poorly formatted documentation when rendered. Excessive indentation may
    /// cause the text to be misinterpreted as a nested list item or code block,
    /// affecting readability and the overall structure of the documentation.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// /// - This is the first item in a list
    /// ///      and this line is overindented.
    /// # fn foo() {}
    /// ```
    ///
    /// Fixes this into:
    /// ```no_run
    /// /// - This is the first item in a list
    /// ///   and this line is overindented.
    /// # fn foo() {}
    /// ```
    #[clippy::version = "1.86.0"]
    pub DOC_OVERINDENTED_LIST_ITEMS,
    style,
    "ensure list items are not overindented"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks if the first paragraph in the documentation of items listed in the module page is too long.
    ///
    /// ### Why is this bad?
    /// Documentation will show the first paragraph of the docstring in the summary page of a
    /// module. Having a nice, short summary in the first paragraph is part of writing good docs.
    ///
    /// ### Example
    /// ```no_run
    /// /// A very short summary.
    /// /// A much longer explanation that goes into a lot more detail about
    /// /// how the thing works, possibly with doclinks and so one,
    /// /// and probably spanning a many rows.
    /// struct Foo {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// /// A very short summary.
    /// ///
    /// /// A much longer explanation that goes into a lot more detail about
    /// /// how the thing works, possibly with doclinks and so one,
    /// /// and probably spanning a many rows.
    /// struct Foo {}
    /// ```
    #[clippy::version = "1.82.0"]
    pub TOO_LONG_FIRST_DOC_PARAGRAPH,
    nursery,
    "ensure the first documentation paragraph is short"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks if included files in doc comments are included only for `cfg(doc)`.
    ///
    /// ### Why restrict this?
    /// These files are not useful for compilation but will still be included.
    /// Also, if any of these non-source code file is updated, it will trigger a
    /// recompilation.
    ///
    /// ### Known problems
    ///
    /// Excluding this will currently result in the file being left out if
    /// the item's docs are inlined from another crate. This may be fixed in a
    /// future version of rustdoc.
    ///
    /// ### Example
    /// ```ignore
    /// #![doc = include_str!("some_file.md")]
    /// ```
    /// Use instead:
    /// ```no_run
    /// #![cfg_attr(doc, doc = include_str!("some_file.md"))]
    /// ```
    #[clippy::version = "1.85.0"]
    pub DOC_INCLUDE_WITHOUT_CFG,
    restriction,
    "check if files included in documentation are behind `cfg(doc)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if a link reference definition appears at the start of a
    /// list item or quote.
    ///
    /// ### Why is this bad?
    /// This is probably intended as an intra-doc link. If it is really
    /// supposed to be a reference definition, it can be written outside
    /// of the list item or quote.
    ///
    /// ### Example
    /// ```no_run
    /// //! - [link]: description
    /// ```
    /// Use instead:
    /// ```no_run
    /// //! - [link][]: description (for intra-doc link)
    /// //!
    /// //! [link]: destination (for link reference definition)
    /// ```
    #[clippy::version = "1.85.0"]
    pub DOC_NESTED_REFDEFS,
    suspicious,
    "link reference defined in list item or quote"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects doc comment linebreaks that use double spaces to separate lines, instead of back-slash (`\`).
    ///
    /// ### Why is this bad?
    /// Double spaces, when used as doc comment linebreaks, can be difficult to see, and may
    /// accidentally be removed during automatic formatting or manual refactoring. The use of a back-slash (`\`)
    /// is clearer in this regard.
    ///
    /// ### Example
    /// The two replacement dots in this example represent a double space.
    /// ```no_run
    /// /// This command takes two numbers as inputs and··
    /// /// adds them together, and then returns the result.
    /// fn add(l: i32, r: i32) -> i32 {
    ///     l + r
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// /// This command takes two numbers as inputs and\
    /// /// adds them together, and then returns the result.
    /// fn add(l: i32, r: i32) -> i32 {
    ///     l + r
    /// }
    /// ```
    #[clippy::version = "1.87.0"]
    pub DOC_COMMENT_DOUBLE_SPACE_LINEBREAKS,
    pedantic,
    "double-space used for doc comment linebreak instead of `\\`"
}

pub struct Documentation {
    valid_idents: FxHashSet<String>,
    check_private_items: bool,
}

impl Documentation {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            valid_idents: conf.doc_valid_idents.iter().cloned().collect(),
            check_private_items: conf.check_private_items,
        }
    }
}

impl_lint_pass!(Documentation => [
    DOC_LINK_CODE,
    DOC_LINK_WITH_QUOTES,
    DOC_MARKDOWN,
    DOC_NESTED_REFDEFS,
    MISSING_SAFETY_DOC,
    MISSING_ERRORS_DOC,
    MISSING_PANICS_DOC,
    NEEDLESS_DOCTEST_MAIN,
    TEST_ATTR_IN_DOCTEST,
    UNNECESSARY_SAFETY_DOC,
    SUSPICIOUS_DOC_COMMENTS,
    EMPTY_DOCS,
    DOC_LAZY_CONTINUATION,
    DOC_OVERINDENTED_LIST_ITEMS,
    TOO_LONG_FIRST_DOC_PARAGRAPH,
    DOC_INCLUDE_WITHOUT_CFG,
    DOC_COMMENT_DOUBLE_SPACE_LINEBREAKS
]);

impl EarlyLintPass for Documentation {
    fn check_attributes(&mut self, cx: &EarlyContext<'_>, attrs: &[rustc_ast::Attribute]) {
        include_in_doc_without_cfg::check(cx, attrs);
    }
}

impl<'tcx> LateLintPass<'tcx> for Documentation {
    fn check_attributes(&mut self, cx: &LateContext<'tcx>, attrs: &'tcx [Attribute]) {
        let Some(headers) = check_attrs(cx, &self.valid_idents, attrs) else {
            return;
        };

        match cx.tcx.hir_node(cx.last_node_with_lint_attrs) {
            Node::Item(item) => {
                too_long_first_doc_paragraph::check(
                    cx,
                    item,
                    attrs,
                    headers.first_paragraph_len,
                    self.check_private_items,
                );
                match item.kind {
                    ItemKind::Fn { sig, body, .. } => {
                        if !(is_entrypoint_fn(cx, item.owner_id.to_def_id())
                            || item.span.in_external_macro(cx.tcx.sess.source_map()))
                        {
                            missing_headers::check(
                                cx,
                                item.owner_id,
                                sig,
                                headers,
                                Some(body),
                                self.check_private_items,
                            );
                        }
                    },
                    ItemKind::Trait(_, unsafety, ..) => match (headers.safety, unsafety) {
                        (false, Safety::Unsafe) => span_lint(
                            cx,
                            MISSING_SAFETY_DOC,
                            cx.tcx.def_span(item.owner_id),
                            "docs for unsafe trait missing `# Safety` section",
                        ),
                        (true, Safety::Safe) => span_lint(
                            cx,
                            UNNECESSARY_SAFETY_DOC,
                            cx.tcx.def_span(item.owner_id),
                            "docs for safe trait have unnecessary `# Safety` section",
                        ),
                        _ => (),
                    },
                    _ => (),
                }
            },
            Node::TraitItem(trait_item) => {
                if let TraitItemKind::Fn(sig, ..) = trait_item.kind
                    && !trait_item.span.in_external_macro(cx.tcx.sess.source_map())
                {
                    missing_headers::check(cx, trait_item.owner_id, sig, headers, None, self.check_private_items);
                }
            },
            Node::ImplItem(impl_item) => {
                if let ImplItemKind::Fn(sig, body_id) = impl_item.kind
                    && !impl_item.span.in_external_macro(cx.tcx.sess.source_map())
                    && !is_trait_impl_item(cx, impl_item.hir_id())
                {
                    missing_headers::check(
                        cx,
                        impl_item.owner_id,
                        sig,
                        headers,
                        Some(body_id),
                        self.check_private_items,
                    );
                }
            },
            _ => {},
        }
    }
}

#[derive(Copy, Clone)]
struct Fragments<'a> {
    doc: &'a str,
    fragments: &'a [DocFragment],
}

impl Fragments<'_> {
    fn span(self, cx: &LateContext<'_>, range: Range<usize>) -> Option<Span> {
        source_span_for_markdown_range(cx.tcx, self.doc, &range, self.fragments)
    }
}

#[derive(Copy, Clone, Default)]
struct DocHeaders {
    safety: bool,
    errors: bool,
    panics: bool,
    first_paragraph_len: usize,
}

/// Does some pre-processing on raw, desugared `#[doc]` attributes such as parsing them and
/// then delegates to `check_doc`.
/// Some lints are already checked here if they can work with attributes directly and don't need
/// to work with markdown.
/// Others are checked elsewhere, e.g. in `check_doc` if they need access to markdown, or
/// back in the various late lint pass methods if they need the final doc headers, like "Safety" or
/// "Panics" sections.
fn check_attrs(cx: &LateContext<'_>, valid_idents: &FxHashSet<String>, attrs: &[Attribute]) -> Option<DocHeaders> {
    /// We don't want the parser to choke on intra doc links. Since we don't
    /// actually care about rendering them, just pretend that all broken links
    /// point to a fake address.
    #[expect(clippy::unnecessary_wraps)] // we're following a type signature
    fn fake_broken_link_callback<'a>(_: BrokenLink<'_>) -> Option<(CowStr<'a>, CowStr<'a>)> {
        Some(("fake".into(), "fake".into()))
    }

    if suspicious_doc_comments::check(cx, attrs) || is_doc_hidden(attrs) {
        return None;
    }

    let (fragments, _) = attrs_to_doc_fragments(
        attrs.iter().filter_map(|attr| {
            if attr.doc_str_and_comment_kind().is_none() || attr.span().in_external_macro(cx.sess().source_map()) {
                None
            } else {
                Some((attr, None))
            }
        }),
        true,
    );
    let mut doc = fragments.iter().fold(String::new(), |mut acc, fragment| {
        add_doc_fragment(&mut acc, fragment);
        acc
    });
    doc.pop();

    if doc.trim().is_empty() {
        if let Some(span) = span_of_fragments(&fragments) {
            span_lint_and_help(
                cx,
                EMPTY_DOCS,
                span,
                "empty doc comment",
                None,
                "consider removing or filling it",
            );
        }
        return Some(DocHeaders::default());
    }

    let mut cb = fake_broken_link_callback;

    check_for_code_clusters(
        cx,
        pulldown_cmark::Parser::new_with_broken_link_callback(
            &doc,
            main_body_opts() - Options::ENABLE_SMART_PUNCTUATION,
            Some(&mut cb),
        )
        .into_offset_iter(),
        &doc,
        Fragments {
            doc: &doc,
            fragments: &fragments,
        },
    );

    // disable smart punctuation to pick up ['link'] more easily
    let opts = main_body_opts() - Options::ENABLE_SMART_PUNCTUATION;
    let parser = pulldown_cmark::Parser::new_with_broken_link_callback(&doc, opts, Some(&mut cb));

    Some(check_doc(
        cx,
        valid_idents,
        parser.into_offset_iter(),
        &doc,
        Fragments {
            doc: &doc,
            fragments: &fragments,
        },
    ))
}

const RUST_CODE: &[&str] = &["rust", "no_run", "should_panic", "compile_fail"];

enum Container {
    Blockquote,
    List(usize),
}

/// Scan the documentation for code links that are back-to-back with code spans.
///
/// This is done separately from the rest of the docs, because that makes it easier to produce
/// the correct messages.
fn check_for_code_clusters<'a, Events: Iterator<Item = (pulldown_cmark::Event<'a>, Range<usize>)>>(
    cx: &LateContext<'_>,
    events: Events,
    doc: &str,
    fragments: Fragments<'_>,
) {
    let mut events = events.peekable();
    let mut code_starts_at = None;
    let mut code_ends_at = None;
    let mut code_includes_link = false;
    while let Some((event, range)) = events.next() {
        match event {
            Start(Link { .. }) if matches!(events.peek(), Some((Code(_), _range))) => {
                if code_starts_at.is_some() {
                    code_ends_at = Some(range.end);
                } else {
                    code_starts_at = Some(range.start);
                }
                code_includes_link = true;
                // skip the nested "code", because we're already handling it here
                let _ = events.next();
            },
            Code(_) => {
                if code_starts_at.is_some() {
                    code_ends_at = Some(range.end);
                } else {
                    code_starts_at = Some(range.start);
                }
            },
            End(TagEnd::Link) => {},
            _ => {
                if let Some(start) = code_starts_at
                    && let Some(end) = code_ends_at
                    && code_includes_link
                    && let Some(span) = fragments.span(cx, start..end)
                {
                    span_lint_and_then(cx, DOC_LINK_CODE, span, "code link adjacent to code text", |diag| {
                        let sugg = format!("<code>{}</code>", doc[start..end].replace('`', ""));
                        diag.span_suggestion_verbose(
                            span,
                            "wrap the entire group in `<code>` tags",
                            sugg,
                            Applicability::MaybeIncorrect,
                        );
                        diag.help("separate code snippets will be shown with a gap");
                    });
                }
                code_includes_link = false;
                code_starts_at = None;
                code_ends_at = None;
            },
        }
    }
}

/// Checks parsed documentation.
/// This walks the "events" (think sections of markdown) produced by `pulldown_cmark`,
/// so lints here will generally access that information.
/// Returns documentation headers -- whether a "Safety", "Errors", "Panic" section was found
#[allow(clippy::too_many_lines)] // Only a big match statement
fn check_doc<'a, Events: Iterator<Item = (pulldown_cmark::Event<'a>, Range<usize>)>>(
    cx: &LateContext<'_>,
    valid_idents: &FxHashSet<String>,
    events: Events,
    doc: &str,
    fragments: Fragments<'_>,
) -> DocHeaders {
    // true if a safety header was found
    let mut headers = DocHeaders::default();
    let mut in_code = false;
    let mut in_link = None;
    let mut in_heading = false;
    let mut in_footnote_definition = false;
    let mut is_rust = false;
    let mut no_test = false;
    let mut ignore = false;
    let mut edition = None;
    let mut ticks_unbalanced = false;
    let mut text_to_check: Vec<(CowStr<'_>, Range<usize>, isize)> = Vec::new();
    let mut paragraph_range = 0..0;
    let mut code_level = 0;
    let mut blockquote_level = 0;
    let mut collected_breaks: Vec<Span> = Vec::new();
    let mut is_first_paragraph = true;

    let mut containers = Vec::new();

    let mut events = events.peekable();

    while let Some((event, range)) = events.next() {
        match event {
            Html(tag) | InlineHtml(tag) => {
                if tag.starts_with("<code") {
                    code_level += 1;
                } else if tag.starts_with("</code") {
                    code_level -= 1;
                } else if tag.starts_with("<blockquote") || tag.starts_with("<q") {
                    blockquote_level += 1;
                } else if tag.starts_with("</blockquote") || tag.starts_with("</q") {
                    blockquote_level -= 1;
                }
            },
            Start(BlockQuote(_)) => {
                blockquote_level += 1;
                containers.push(Container::Blockquote);
                if let Some((next_event, next_range)) = events.peek() {
                    let next_start = match next_event {
                        End(TagEnd::BlockQuote) => next_range.end,
                        _ => next_range.start,
                    };
                    if let Some(refdefrange) = looks_like_refdef(doc, range.start..next_start) &&
                        let Some(refdefspan) = fragments.span(cx, refdefrange.clone())
                    {
                        span_lint_and_then(
                            cx,
                            DOC_NESTED_REFDEFS,
                            refdefspan,
                            "link reference defined in quote",
                            |diag| {
                                diag.span_suggestion_short(
                                    refdefspan.shrink_to_hi(),
                                    "for an intra-doc link, add `[]` between the label and the colon",
                                    "[]",
                                    Applicability::MaybeIncorrect,
                                );
                                diag.help("link definitions are not shown in rendered documentation");
                            }
                        );
                    }
                }
            },
            End(TagEnd::BlockQuote) => {
                blockquote_level -= 1;
                containers.pop();
            },
            Start(CodeBlock(ref kind)) => {
                in_code = true;
                if let CodeBlockKind::Fenced(lang) = kind {
                    for item in lang.split(',') {
                        if item == "ignore" {
                            is_rust = false;
                            break;
                        } else if item == "no_test" {
                            no_test = true;
                        } else if item == "no_run" || item == "compile_fail" {
                            ignore = true;
                        }
                        if let Some(stripped) = item.strip_prefix("edition") {
                            is_rust = true;
                            edition = stripped.parse::<Edition>().ok();
                        } else if item.is_empty() || RUST_CODE.contains(&item) {
                            is_rust = true;
                        }
                    }
                }
            },
            End(TagEnd::CodeBlock) => {
                in_code = false;
                is_rust = false;
                ignore = false;
            },
            Start(Link { dest_url, .. }) => in_link = Some(dest_url),
            End(TagEnd::Link) => in_link = None,
            Start(Heading { .. } | Paragraph | Item) => {
                if let Start(Heading { .. }) = event {
                    in_heading = true;
                }
                if let Start(Item) = event {
                    let indent = if let Some((next_event, next_range)) = events.peek() {
                        let next_start = match next_event {
                            End(TagEnd::Item) => next_range.end,
                            _ => next_range.start,
                        };
                        if let Some(refdefrange) = looks_like_refdef(doc, range.start..next_start) &&
                            let Some(refdefspan) = fragments.span(cx, refdefrange.clone())
                        {
                            span_lint_and_then(
                                cx,
                                DOC_NESTED_REFDEFS,
                                refdefspan,
                                "link reference defined in list item",
                                |diag| {
                                    diag.span_suggestion_short(
                                        refdefspan.shrink_to_hi(),
                                        "for an intra-doc link, add `[]` between the label and the colon",
                                        "[]",
                                        Applicability::MaybeIncorrect,
                                    );
                                    diag.help("link definitions are not shown in rendered documentation");
                                }
                            );
                            refdefrange.start - range.start
                        } else {
                            let mut start = next_range.start;
                            if start > 0 && doc.as_bytes().get(start - 1) == Some(&b'\\') {
                                // backslashes aren't in the event stream...
                                start -= 1;
                            }

                            start.saturating_sub(range.start)
                        }
                    } else {
                        0
                    };
                    containers.push(Container::List(indent));
                }
                ticks_unbalanced = false;
                paragraph_range = range;
                if is_first_paragraph {
                    headers.first_paragraph_len = doc[paragraph_range.clone()].chars().count();
                    is_first_paragraph = false;
                }
            },
            End(TagEnd::Heading(_) | TagEnd::Paragraph | TagEnd::Item) => {
                if let End(TagEnd::Heading(_)) = event {
                    in_heading = false;
                }
                if let End(TagEnd::Item) = event {
                    containers.pop();
                }
                if ticks_unbalanced && let Some(span) = fragments.span(cx, paragraph_range.clone()) {
                    span_lint_and_help(
                        cx,
                        DOC_MARKDOWN,
                        span,
                        "backticks are unbalanced",
                        None,
                        "a backtick may be missing a pair",
                    );
                } else {
                    for (text, range, assoc_code_level) in text_to_check {
                        if let Some(span) = fragments.span(cx, range) {
                            markdown::check(cx, valid_idents, &text, span, assoc_code_level, blockquote_level);
                        }
                    }
                }
                text_to_check = Vec::new();
            },
            Start(FootnoteDefinition(..)) => in_footnote_definition = true,
            End(TagEnd::FootnoteDefinition) => in_footnote_definition = false,
            Start(_) | End(_)  // We don't care about other tags
            | TaskListMarker(_) | Code(_) | Rule | InlineMath(..) | DisplayMath(..) => (),
            SoftBreak | HardBreak => {
                if !containers.is_empty()
                    && let Some((next_event, next_range)) = events.peek()
                    && let Some(next_span) = fragments.span(cx, next_range.clone())
                    && let Some(span) = fragments.span(cx, range.clone())
                    && !in_footnote_definition
                    && !matches!(next_event, End(_))
                {
                    lazy_continuation::check(
                        cx,
                        doc,
                        range.end..next_range.start,
                        Span::new(span.hi(), next_span.lo(), span.ctxt(), span.parent()),
                        &containers[..],
                    );
                }

                if let Some(span) = fragments.span(cx, range.clone())
                    && !span.from_expansion()
                    && let Some(snippet) = snippet_opt(cx, span)
                    && !snippet.trim().starts_with('\\')
                    && event == HardBreak {
                    collected_breaks.push(span);
                }
            },
            Text(text) => {
                paragraph_range.end = range.end;
                let range_ = range.clone();
                ticks_unbalanced |= text.contains('`')
                    && !in_code
                    && doc[range.clone()].bytes().enumerate().any(|(i, c)| {
                        // scan the markdown source code bytes for backquotes that aren't preceded by backslashes
                        // - use bytes, instead of chars, to avoid utf8 decoding overhead (special chars are ascii)
                        // - relevant backquotes are within doc[range], but backslashes are not, because they're not
                        //   actually part of the rendered text (pulldown-cmark doesn't emit any events for escapes)
                        // - if `range_.start + i == 0`, then `range_.start + i - 1 == -1`, and since we're working in
                        //   usize, that would underflow and maybe panic
                        c == b'`' && (range_.start + i == 0 || doc.as_bytes().get(range_.start + i - 1) != Some(&b'\\'))
                    });
                if Some(&text) == in_link.as_ref() || ticks_unbalanced {
                    // Probably a link of the form `<http://example.com>`
                    // Which are represented as a link to "http://example.com" with
                    // text "http://example.com" by pulldown-cmark
                    continue;
                }
                let trimmed_text = text.trim();
                headers.safety |= in_heading && trimmed_text == "Safety";
                headers.safety |= in_heading && trimmed_text == "SAFETY";
                headers.safety |= in_heading && trimmed_text == "Implementation safety";
                headers.safety |= in_heading && trimmed_text == "Implementation Safety";
                headers.errors |= in_heading && trimmed_text == "Errors";
                headers.panics |= in_heading && trimmed_text == "Panics";
                if in_code {
                    if is_rust && !no_test {
                        let edition = edition.unwrap_or_else(|| cx.tcx.sess.edition());
                        needless_doctest_main::check(cx, &text, edition, range.clone(), fragments, ignore);
                    }
                } else {
                    if in_link.is_some() {
                        link_with_quotes::check(cx, trimmed_text, range.clone(), fragments);
                    }
                    if let Some(link) = in_link.as_ref()
                        && let Ok(url) = Url::parse(link)
                        && (url.scheme() == "https" || url.scheme() == "http")
                    {
                        // Don't check the text associated with external URLs
                        continue;
                    }
                    text_to_check.push((text, range, code_level));
                }
            }
            FootnoteReference(_) => {}
        }
    }

    doc_comment_double_space_linebreaks::check(cx, &collected_breaks);

    headers
}

#[expect(clippy::range_plus_one)] // inclusive ranges aren't the same type
fn looks_like_refdef(doc: &str, range: Range<usize>) -> Option<Range<usize>> {
    if range.end < range.start {
        return None;
    }

    let offset = range.start;
    let mut iterator = doc.as_bytes()[range].iter().copied().enumerate();
    let mut start = None;
    while let Some((i, byte)) = iterator.next() {
        match byte {
            b'\\' => {
                iterator.next();
            },
            b'[' => {
                start = Some(i + offset);
            },
            b']' if let Some(start) = start => {
                return Some(start..i + offset + 1);
            },
            _ => {},
        }
    }
    None
}
