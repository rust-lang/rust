mod lazy_continuation;
use clippy_utils::attrs::is_doc_hidden;
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::macros::{is_panic, root_macro_call_first_node};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::Visitable;
use clippy_utils::{in_constant, is_entrypoint_fn, is_trait_impl_item, method_chain_args};
use pulldown_cmark::Event::{
    Code, End, FootnoteReference, HardBreak, Html, Rule, SoftBreak, Start, TaskListMarker, Text,
};
use pulldown_cmark::Tag::{BlockQuote, CodeBlock, FootnoteDefinition, Heading, Item, Link, Paragraph};
use pulldown_cmark::{BrokenLink, CodeBlockKind, CowStr, Options};
use rustc_ast::ast::Attribute;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{AnonConst, Expr, ImplItemKind, ItemKind, Node, Safety, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::nested_filter;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_resolve::rustdoc::{
    add_doc_fragment, attrs_to_doc_fragments, main_body_opts, source_span_for_markdown_range, span_of_fragments,
    DocFragment,
};
use rustc_session::impl_lint_pass;
use rustc_span::edition::Edition;
use rustc_span::{sym, Span};
use std::ops::Range;
use url::Url;

mod link_with_quotes;
mod markdown;
mod missing_headers;
mod needless_doctest_main;
mod suspicious_doc_comments;

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
    /// ```rs
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

#[derive(Clone)]
pub struct Documentation {
    valid_idents: FxHashSet<String>,
    check_private_items: bool,
}

impl Documentation {
    pub fn new(valid_idents: &[String], check_private_items: bool) -> Self {
        Self {
            valid_idents: valid_idents.iter().cloned().collect(),
            check_private_items,
        }
    }
}

impl_lint_pass!(Documentation => [
    DOC_LINK_WITH_QUOTES,
    DOC_MARKDOWN,
    MISSING_SAFETY_DOC,
    MISSING_ERRORS_DOC,
    MISSING_PANICS_DOC,
    NEEDLESS_DOCTEST_MAIN,
    TEST_ATTR_IN_DOCTEST,
    UNNECESSARY_SAFETY_DOC,
    SUSPICIOUS_DOC_COMMENTS,
    EMPTY_DOCS,
    DOC_LAZY_CONTINUATION,
]);

impl<'tcx> LateLintPass<'tcx> for Documentation {
    fn check_attributes(&mut self, cx: &LateContext<'tcx>, attrs: &'tcx [Attribute]) {
        let Some(headers) = check_attrs(cx, &self.valid_idents, attrs) else {
            return;
        };

        match cx.tcx.hir_node(cx.last_node_with_lint_attrs) {
            Node::Item(item) => match item.kind {
                ItemKind::Fn(sig, _, body_id) => {
                    if !(is_entrypoint_fn(cx, item.owner_id.to_def_id()) || in_external_macro(cx.tcx.sess, item.span)) {
                        let body = cx.tcx.hir().body(body_id);

                        let panic_info = FindPanicUnwrap::find_span(cx, cx.tcx.typeck(item.owner_id), body.value);
                        missing_headers::check(
                            cx,
                            item.owner_id,
                            sig,
                            headers,
                            Some(body_id),
                            panic_info,
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
            },
            Node::TraitItem(trait_item) => {
                if let TraitItemKind::Fn(sig, ..) = trait_item.kind
                    && !in_external_macro(cx.tcx.sess, trait_item.span)
                {
                    missing_headers::check(
                        cx,
                        trait_item.owner_id,
                        sig,
                        headers,
                        None,
                        None,
                        self.check_private_items,
                    );
                }
            },
            Node::ImplItem(impl_item) => {
                if let ImplItemKind::Fn(sig, body_id) = impl_item.kind
                    && !in_external_macro(cx.tcx.sess, impl_item.span)
                    && !is_trait_impl_item(cx, impl_item.hir_id())
                {
                    let body = cx.tcx.hir().body(body_id);

                    let panic_span = FindPanicUnwrap::find_span(cx, cx.tcx.typeck(impl_item.owner_id), body.value);
                    missing_headers::check(
                        cx,
                        impl_item.owner_id,
                        sig,
                        headers,
                        Some(body_id),
                        panic_span,
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

    if is_doc_hidden(attrs) {
        return None;
    }

    suspicious_doc_comments::check(cx, attrs);

    let (fragments, _) = attrs_to_doc_fragments(
        attrs.iter().filter_map(|attr| {
            if in_external_macro(cx.sess(), attr.span) {
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

    // disable smart punctuation to pick up ['link'] more easily
    let opts = main_body_opts() - Options::ENABLE_SMART_PUNCTUATION;
    let parser = pulldown_cmark::Parser::new_with_broken_link_callback(&doc, opts, Some(&mut cb));

    Some(check_doc(
        cx,
        valid_idents,
        parser.into_offset_iter(),
        &doc,
        Fragments {
            fragments: &fragments,
            doc: &doc,
        },
    ))
}

const RUST_CODE: &[&str] = &["rust", "no_run", "should_panic", "compile_fail"];

enum Container {
    Blockquote,
    List(usize),
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

    let mut containers = Vec::new();

    let mut events = events.peekable();

    while let Some((event, range)) = events.next() {
        match event {
            Html(tag) => {
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
            Start(BlockQuote) => {
                blockquote_level += 1;
                containers.push(Container::Blockquote);
            },
            End(BlockQuote) => {
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
            End(CodeBlock(_)) => {
                in_code = false;
                is_rust = false;
                ignore = false;
            },
            Start(Link(_, url, _)) => in_link = Some(url),
            End(Link(..)) => in_link = None,
            Start(Heading(_, _, _) | Paragraph | Item) => {
                if let Start(Heading(_, _, _)) = event {
                    in_heading = true;
                }
                if let Start(Item) = event {
                    if let Some((_next_event, next_range)) = events.peek() {
                        containers.push(Container::List(next_range.start - range.start));
                    } else {
                        containers.push(Container::List(0));
                    }
                }
                ticks_unbalanced = false;
                paragraph_range = range;
            },
            End(Heading(_, _, _) | Paragraph | Item) => {
                if let End(Heading(_, _, _)) = event {
                    in_heading = false;
                }
                if let End(Item) = event {
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
            End(FootnoteDefinition(..)) => in_footnote_definition = false,
            Start(_tag) | End(_tag) => (), // We don't care about other tags
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
                        span,
                    );
                }
            },
            TaskListMarker(_) | Code(_) | Rule => (),
            FootnoteReference(text) | Text(text) => {
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
            },
        }
    }
    headers
}

struct FindPanicUnwrap<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    is_const: bool,
    panic_span: Option<Span>,
    typeck_results: &'tcx ty::TypeckResults<'tcx>,
}

impl<'a, 'tcx> FindPanicUnwrap<'a, 'tcx> {
    pub fn find_span(
        cx: &'a LateContext<'tcx>,
        typeck_results: &'tcx ty::TypeckResults<'tcx>,
        body: impl Visitable<'tcx>,
    ) -> Option<(Span, bool)> {
        let mut vis = Self {
            cx,
            is_const: false,
            panic_span: None,
            typeck_results,
        };
        body.visit(&mut vis);
        vis.panic_span.map(|el| (el, vis.is_const))
    }
}

impl<'a, 'tcx> Visitor<'tcx> for FindPanicUnwrap<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.panic_span.is_some() {
            return;
        }

        if let Some(macro_call) = root_macro_call_first_node(self.cx, expr) {
            if is_panic(self.cx, macro_call.def_id)
                || matches!(
                    self.cx.tcx.item_name(macro_call.def_id).as_str(),
                    "assert" | "assert_eq" | "assert_ne"
                )
            {
                self.is_const = in_constant(self.cx, expr.hir_id);
                self.panic_span = Some(macro_call.span);
            }
        }

        // check for `unwrap` and `expect` for both `Option` and `Result`
        if let Some(arglists) = method_chain_args(expr, &["unwrap"]).or(method_chain_args(expr, &["expect"])) {
            let receiver_ty = self.typeck_results.expr_ty(arglists[0].0).peel_refs();
            if is_type_diagnostic_item(self.cx, receiver_ty, sym::Option)
                || is_type_diagnostic_item(self.cx, receiver_ty, sym::Result)
            {
                self.panic_span = Some(expr.span);
            }
        }

        // and check sub-expressions
        intravisit::walk_expr(self, expr);
    }

    // Panics in const blocks will cause compilation to fail.
    fn visit_anon_const(&mut self, _: &'tcx AnonConst) {}

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}
