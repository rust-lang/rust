use clippy_utils::attrs::is_doc_hidden;
use clippy_utils::diagnostics::{span_lint, span_lint_and_help, span_lint_and_note, span_lint_and_sugg};
use clippy_utils::source::{first_line_of_span, snippet_with_applicability};
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use clippy_utils::{is_entrypoint_fn, is_expn_of, match_panic_def_id, method_chain_args, return_ty};
use if_chain::if_chain;
use itertools::Itertools;
use rustc_ast::ast::{Async, AttrKind, Attribute, Fn, FnRetTy, ItemKind};
use rustc_ast::token::CommentKind;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_errors::emitter::EmitterWriter;
use rustc_errors::{Applicability, Handler};
use rustc_hir as hir;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{AnonConst, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_parse::maybe_new_parser_from_source_str;
use rustc_parse::parser::ForceCollect;
use rustc_session::parse::ParseSess;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::LocalDefId;
use rustc_span::edition::Edition;
use rustc_span::source_map::{BytePos, FilePathMapping, MultiSpan, SourceMap, Span};
use rustc_span::{sym, FileName, Pos};
use std::io;
use std::ops::Range;
use std::thread;
use url::Url;

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
    /// ```rust
    /// /// Do something with the foo_bar parameter. See also
    /// /// that::other::module::foo.
    /// // ^ `foo_bar` and `that::other::module::foo` should be ticked.
    /// fn doit(foo_bar: usize) {}
    /// ```
    ///
    /// ```rust
    /// // Link text with `[]` brackets should be written as following:
    /// /// Consume the array and return the inner
    /// /// [`SmallVec<[T; INLINE_CAPACITY]>`][SmallVec].
    /// /// [SmallVec]: SmallVec
    /// fn main() {}
    /// ```
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
    /// ```rust
    ///# type Universe = ();
    /// /// This function should really be documented
    /// pub unsafe fn start_apocalypse(u: &mut Universe) {
    ///     unimplemented!();
    /// }
    /// ```
    ///
    /// At least write a line about safety:
    ///
    /// ```rust
    ///# type Universe = ();
    /// /// # Safety
    /// ///
    /// /// This function should not be called before the horsemen are ready.
    /// pub unsafe fn start_apocalypse(u: &mut Universe) {
    ///     unimplemented!();
    /// }
    /// ```
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
    /// ```rust
    ///# use std::io;
    /// /// # Errors
    /// ///
    /// /// Will return `Err` if `filename` does not exist or the user does not have
    /// /// permission to read it.
    /// pub fn read(filename: String) -> io::Result<String> {
    ///     unimplemented!();
    /// }
    /// ```
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
    /// ```rust
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
    /// ``````rust
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
    /// ``````
    pub NEEDLESS_DOCTEST_MAIN,
    style,
    "presence of `fn main() {` in code examples"
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct DocMarkdown {
    valid_idents: FxHashSet<String>,
    in_trait_impl: bool,
}

impl DocMarkdown {
    pub fn new(valid_idents: FxHashSet<String>) -> Self {
        Self {
            valid_idents,
            in_trait_impl: false,
        }
    }
}

impl_lint_pass!(DocMarkdown =>
    [DOC_MARKDOWN, MISSING_SAFETY_DOC, MISSING_ERRORS_DOC, MISSING_PANICS_DOC, NEEDLESS_DOCTEST_MAIN]
);

impl<'tcx> LateLintPass<'tcx> for DocMarkdown {
    fn check_crate(&mut self, cx: &LateContext<'tcx>) {
        let attrs = cx.tcx.hir().attrs(hir::CRATE_HIR_ID);
        check_attrs(cx, &self.valid_idents, attrs);
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        let attrs = cx.tcx.hir().attrs(item.hir_id());
        let headers = check_attrs(cx, &self.valid_idents, attrs);
        match item.kind {
            hir::ItemKind::Fn(ref sig, _, body_id) => {
                if !(is_entrypoint_fn(cx, item.def_id.to_def_id()) || in_external_macro(cx.tcx.sess, item.span)) {
                    let body = cx.tcx.hir().body(body_id);
                    let mut fpu = FindPanicUnwrap {
                        cx,
                        typeck_results: cx.tcx.typeck(item.def_id),
                        panic_span: None,
                    };
                    fpu.visit_expr(&body.value);
                    lint_for_missing_headers(cx, item.def_id, item.span, sig, headers, Some(body_id), fpu.panic_span);
                }
            },
            hir::ItemKind::Impl(ref impl_) => {
                self.in_trait_impl = impl_.of_trait.is_some();
            },
            hir::ItemKind::Trait(_, unsafety, ..) => {
                if !headers.safety && unsafety == hir::Unsafety::Unsafe {
                    span_lint(
                        cx,
                        MISSING_SAFETY_DOC,
                        item.span,
                        "docs for unsafe trait missing `# Safety` section",
                    );
                }
            },
            _ => (),
        }
    }

    fn check_item_post(&mut self, _cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if let hir::ItemKind::Impl { .. } = item.kind {
            self.in_trait_impl = false;
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
        let attrs = cx.tcx.hir().attrs(item.hir_id());
        let headers = check_attrs(cx, &self.valid_idents, attrs);
        if let hir::TraitItemKind::Fn(ref sig, ..) = item.kind {
            if !in_external_macro(cx.tcx.sess, item.span) {
                lint_for_missing_headers(cx, item.def_id, item.span, sig, headers, None, None);
            }
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
        let attrs = cx.tcx.hir().attrs(item.hir_id());
        let headers = check_attrs(cx, &self.valid_idents, attrs);
        if self.in_trait_impl || in_external_macro(cx.tcx.sess, item.span) {
            return;
        }
        if let hir::ImplItemKind::Fn(ref sig, body_id) = item.kind {
            let body = cx.tcx.hir().body(body_id);
            let mut fpu = FindPanicUnwrap {
                cx,
                typeck_results: cx.tcx.typeck(item.def_id),
                panic_span: None,
            };
            fpu.visit_expr(&body.value);
            lint_for_missing_headers(cx, item.def_id, item.span, sig, headers, Some(body_id), fpu.panic_span);
        }
    }
}

fn lint_for_missing_headers<'tcx>(
    cx: &LateContext<'tcx>,
    def_id: LocalDefId,
    span: impl Into<MultiSpan> + Copy,
    sig: &hir::FnSig<'_>,
    headers: DocHeaders,
    body_id: Option<hir::BodyId>,
    panic_span: Option<Span>,
) {
    if !cx.access_levels.is_exported(def_id) {
        return; // Private functions do not require doc comments
    }

    // do not lint if any parent has `#[doc(hidden)]` attribute (#7347)
    if cx
        .tcx
        .hir()
        .parent_iter(cx.tcx.hir().local_def_id_to_hir_id(def_id))
        .any(|(id, _node)| is_doc_hidden(cx.tcx.hir().attrs(id)))
    {
        return;
    }

    if !headers.safety && sig.header.unsafety == hir::Unsafety::Unsafe {
        span_lint(
            cx,
            MISSING_SAFETY_DOC,
            span,
            "unsafe function's docs miss `# Safety` section",
        );
    }
    if !headers.panics && panic_span.is_some() {
        span_lint_and_note(
            cx,
            MISSING_PANICS_DOC,
            span,
            "docs for function which may panic missing `# Panics` section",
            panic_span,
            "first possible panic found here",
        );
    }
    if !headers.errors {
        let hir_id = cx.tcx.hir().local_def_id_to_hir_id(def_id);
        if is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym::Result) {
            span_lint(
                cx,
                MISSING_ERRORS_DOC,
                span,
                "docs for function returning `Result` missing `# Errors` section",
            );
        } else {
            if_chain! {
                if let Some(body_id) = body_id;
                if let Some(future) = cx.tcx.lang_items().future_trait();
                let typeck = cx.tcx.typeck_body(body_id);
                let body = cx.tcx.hir().body(body_id);
                let ret_ty = typeck.expr_ty(&body.value);
                if implements_trait(cx, ret_ty, future, &[]);
                if let ty::Opaque(_, subs) = ret_ty.kind();
                if let Some(gen) = subs.types().next();
                if let ty::Generator(_, subs, _) = gen.kind();
                if is_type_diagnostic_item(cx, subs.as_generator().return_ty(), sym::Result);
                then {
                    span_lint(
                        cx,
                        MISSING_ERRORS_DOC,
                        span,
                        "docs for function returning `Result` missing `# Errors` section",
                    );
                }
            }
        }
    }
}

/// Cleanup documentation decoration.
///
/// We can't use `rustc_ast::attr::AttributeMethods::with_desugared_doc` or
/// `rustc_ast::parse::lexer::comments::strip_doc_comment_decoration` because we
/// need to keep track of
/// the spans but this function is inspired from the later.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn strip_doc_comment_decoration(doc: &str, comment_kind: CommentKind, span: Span) -> (String, Vec<(usize, Span)>) {
    // one-line comments lose their prefix
    if comment_kind == CommentKind::Line {
        let mut doc = doc.to_owned();
        doc.push('\n');
        let len = doc.len();
        // +3 skips the opening delimiter
        return (doc, vec![(len, span.with_lo(span.lo() + BytePos(3)))]);
    }

    let mut sizes = vec![];
    let mut contains_initial_stars = false;
    for line in doc.lines() {
        let offset = line.as_ptr() as usize - doc.as_ptr() as usize;
        debug_assert_eq!(offset as u32 as usize, offset);
        contains_initial_stars |= line.trim_start().starts_with('*');
        // +1 adds the newline, +3 skips the opening delimiter
        sizes.push((line.len() + 1, span.with_lo(span.lo() + BytePos(3 + offset as u32))));
    }
    if !contains_initial_stars {
        return (doc.to_string(), sizes);
    }
    // remove the initial '*'s if any
    let mut no_stars = String::with_capacity(doc.len());
    for line in doc.lines() {
        let mut chars = line.chars();
        for c in &mut chars {
            if c.is_whitespace() {
                no_stars.push(c);
            } else {
                no_stars.push(if c == '*' { ' ' } else { c });
                break;
            }
        }
        no_stars.push_str(chars.as_str());
        no_stars.push('\n');
    }

    (no_stars, sizes)
}

#[derive(Copy, Clone)]
struct DocHeaders {
    safety: bool,
    errors: bool,
    panics: bool,
}

fn check_attrs<'a>(cx: &LateContext<'_>, valid_idents: &FxHashSet<String>, attrs: &'a [Attribute]) -> DocHeaders {
    use pulldown_cmark::{BrokenLink, CowStr, Options};
    /// We don't want the parser to choke on intra doc links. Since we don't
    /// actually care about rendering them, just pretend that all broken links are
    /// point to a fake address.
    #[allow(clippy::unnecessary_wraps)] // we're following a type signature
    fn fake_broken_link_callback<'a>(_: BrokenLink<'_>) -> Option<(CowStr<'a>, CowStr<'a>)> {
        Some(("fake".into(), "fake".into()))
    }

    let mut doc = String::new();
    let mut spans = vec![];

    for attr in attrs {
        if let AttrKind::DocComment(comment_kind, comment) = attr.kind {
            let (comment, current_spans) = strip_doc_comment_decoration(&comment.as_str(), comment_kind, attr.span);
            spans.extend_from_slice(&current_spans);
            doc.push_str(&comment);
        } else if attr.has_name(sym::doc) {
            // ignore mix of sugared and non-sugared doc
            // don't trigger the safety or errors check
            return DocHeaders {
                safety: true,
                errors: true,
                panics: true,
            };
        }
    }

    let mut current = 0;
    for &mut (ref mut offset, _) in &mut spans {
        let offset_copy = *offset;
        *offset = current;
        current += offset_copy;
    }

    if doc.is_empty() {
        return DocHeaders {
            safety: false,
            errors: false,
            panics: false,
        };
    }

    let mut cb = fake_broken_link_callback;

    let parser =
        pulldown_cmark::Parser::new_with_broken_link_callback(&doc, Options::empty(), Some(&mut cb)).into_offset_iter();
    // Iterate over all `Events` and combine consecutive events into one
    let events = parser.coalesce(|previous, current| {
        use pulldown_cmark::Event::Text;

        let previous_range = previous.1;
        let current_range = current.1;

        match (previous.0, current.0) {
            (Text(previous), Text(current)) => {
                let mut previous = previous.to_string();
                previous.push_str(&current);
                Ok((Text(previous.into()), previous_range))
            },
            (previous, current) => Err(((previous, previous_range), (current, current_range))),
        }
    });
    check_doc(cx, valid_idents, events, &spans)
}

const RUST_CODE: &[&str] = &["rust", "no_run", "should_panic", "compile_fail"];

fn check_doc<'a, Events: Iterator<Item = (pulldown_cmark::Event<'a>, Range<usize>)>>(
    cx: &LateContext<'_>,
    valid_idents: &FxHashSet<String>,
    events: Events,
    spans: &[(usize, Span)],
) -> DocHeaders {
    // true if a safety header was found
    use pulldown_cmark::Event::{
        Code, End, FootnoteReference, HardBreak, Html, Rule, SoftBreak, Start, TaskListMarker, Text,
    };
    use pulldown_cmark::Tag::{CodeBlock, Heading, Item, Link, Paragraph};
    use pulldown_cmark::{CodeBlockKind, CowStr};

    let mut headers = DocHeaders {
        safety: false,
        errors: false,
        panics: false,
    };
    let mut in_code = false;
    let mut in_link = None;
    let mut in_heading = false;
    let mut is_rust = false;
    let mut edition = None;
    let mut ticks_unbalanced = false;
    let mut text_to_check: Vec<(CowStr<'_>, Span)> = Vec::new();
    let mut paragraph_span = spans.get(0).expect("function isn't called if doc comment is empty").1;
    for (event, range) in events {
        match event {
            Start(CodeBlock(ref kind)) => {
                in_code = true;
                if let CodeBlockKind::Fenced(lang) = kind {
                    for item in lang.split(',') {
                        if item == "ignore" {
                            is_rust = false;
                            break;
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
            },
            Start(Link(_, url, _)) => in_link = Some(url),
            End(Link(..)) => in_link = None,
            Start(Heading(_) | Paragraph | Item) => {
                if let Start(Heading(_)) = event {
                    in_heading = true;
                }
                ticks_unbalanced = false;
                let (_, span) = get_current_span(spans, range.start);
                paragraph_span = first_line_of_span(cx, span);
            },
            End(Heading(_) | Paragraph | Item) => {
                if let End(Heading(_)) = event {
                    in_heading = false;
                }
                if ticks_unbalanced {
                    span_lint_and_help(
                        cx,
                        DOC_MARKDOWN,
                        paragraph_span,
                        "backticks are unbalanced",
                        None,
                        "a backtick may be missing a pair",
                    );
                } else {
                    for (text, span) in text_to_check {
                        check_text(cx, valid_idents, &text, span);
                    }
                }
                text_to_check = Vec::new();
            },
            Start(_tag) | End(_tag) => (), // We don't care about other tags
            Html(_html) => (),             // HTML is weird, just ignore it
            SoftBreak | HardBreak | TaskListMarker(_) | Code(_) | Rule => (),
            FootnoteReference(text) | Text(text) => {
                let (begin, span) = get_current_span(spans, range.start);
                paragraph_span = paragraph_span.with_hi(span.hi());
                ticks_unbalanced |= text.contains('`') && !in_code;
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
                    if is_rust {
                        let edition = edition.unwrap_or_else(|| cx.tcx.sess.edition());
                        check_code(cx, &text, edition, span);
                    }
                } else {
                    // Adjust for the beginning of the current `Event`
                    let span = span.with_lo(span.lo() + BytePos::from_usize(range.start - begin));
                    text_to_check.push((text, span));
                }
            },
        }
    }
    headers
}

fn get_current_span(spans: &[(usize, Span)], idx: usize) -> (usize, Span) {
    let index = match spans.binary_search_by(|c| c.0.cmp(&idx)) {
        Ok(o) => o,
        Err(e) => e - 1,
    };
    spans[index]
}

fn check_code(cx: &LateContext<'_>, text: &str, edition: Edition, span: Span) {
    fn has_needless_main(code: String, edition: Edition) -> bool {
        rustc_driver::catch_fatal_errors(|| {
            rustc_span::create_session_globals_then(edition, || {
                let filename = FileName::anon_source_code(&code);

                let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
                let emitter = EmitterWriter::new(Box::new(io::sink()), None, false, false, false, None, false);
                let handler = Handler::with_emitter(false, None, Box::new(emitter));
                let sess = ParseSess::with_span_handler(handler, sm);

                let mut parser = match maybe_new_parser_from_source_str(&sess, filename, code) {
                    Ok(p) => p,
                    Err(errs) => {
                        for mut err in errs {
                            err.cancel();
                        }
                        return false;
                    },
                };

                let mut relevant_main_found = false;
                loop {
                    match parser.parse_item(ForceCollect::No) {
                        Ok(Some(item)) => match &item.kind {
                            // Tests with one of these items are ignored
                            ItemKind::Static(..)
                            | ItemKind::Const(..)
                            | ItemKind::ExternCrate(..)
                            | ItemKind::ForeignMod(..) => return false,
                            // We found a main function ...
                            ItemKind::Fn(box Fn { sig, body: Some(block), .. }) if item.ident.name == sym::main => {
                                let is_async = matches!(sig.header.asyncness, Async::Yes { .. });
                                let returns_nothing = match &sig.decl.output {
                                    FnRetTy::Default(..) => true,
                                    FnRetTy::Ty(ty) if ty.kind.is_unit() => true,
                                    FnRetTy::Ty(_) => false,
                                };

                                if returns_nothing && !is_async && !block.stmts.is_empty() {
                                    // This main function should be linted, but only if there are no other functions
                                    relevant_main_found = true;
                                } else {
                                    // This main function should not be linted, we're done
                                    return false;
                                }
                            },
                            // Another function was found; this case is ignored too
                            ItemKind::Fn(..) => return false,
                            _ => {},
                        },
                        Ok(None) => break,
                        Err(mut e) => {
                            e.cancel();
                            return false;
                        },
                    }
                }

                relevant_main_found
            })
        })
        .ok()
        .unwrap_or_default()
    }

    // Because of the global session, we need to create a new session in a different thread with
    // the edition we need.
    let text = text.to_owned();
    if thread::spawn(move || has_needless_main(text, edition))
        .join()
        .expect("thread::spawn failed")
    {
        span_lint(cx, NEEDLESS_DOCTEST_MAIN, span, "needless `fn main` in doctest");
    }
}

fn check_text(cx: &LateContext<'_>, valid_idents: &FxHashSet<String>, text: &str, span: Span) {
    for word in text.split(|c: char| c.is_whitespace() || c == '\'') {
        // Trim punctuation as in `some comment (see foo::bar).`
        //                                                   ^^
        // Or even as in `_foo bar_` which is emphasized. Also preserve `::` as a prefix/suffix.
        let mut word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != ':');

        // Remove leading or trailing single `:` which may be part of a sentence.
        if word.starts_with(':') && !word.starts_with("::") {
            word = word.trim_start_matches(':');
        }
        if word.ends_with(':') && !word.ends_with("::") {
            word = word.trim_end_matches(':');
        }

        if valid_idents.contains(word) || word.chars().all(|c| c == ':') {
            continue;
        }

        // Adjust for the current word
        let offset = word.as_ptr() as usize - text.as_ptr() as usize;
        let span = Span::new(
            span.lo() + BytePos::from_usize(offset),
            span.lo() + BytePos::from_usize(offset + word.len()),
            span.ctxt(),
            span.parent(),
        );

        check_word(cx, word, span);
    }
}

fn check_word(cx: &LateContext<'_>, word: &str, span: Span) {
    /// Checks if a string is camel-case, i.e., contains at least two uppercase
    /// letters (`Clippy` is ok) and one lower-case letter (`NASA` is ok).
    /// Plurals are also excluded (`IDs` is ok).
    fn is_camel_case(s: &str) -> bool {
        if s.starts_with(|c: char| c.is_digit(10)) {
            return false;
        }

        let s = s.strip_suffix('s').unwrap_or(s);

        s.chars().all(char::is_alphanumeric)
            && s.chars().filter(|&c| c.is_uppercase()).take(2).count() > 1
            && s.chars().filter(|&c| c.is_lowercase()).take(1).count() > 0
    }

    fn has_underscore(s: &str) -> bool {
        s != "_" && !s.contains("\\_") && s.contains('_')
    }

    fn has_hyphen(s: &str) -> bool {
        s != "-" && s.contains('-')
    }

    if let Ok(url) = Url::parse(word) {
        // try to get around the fact that `foo::bar` parses as a valid URL
        if !url.cannot_be_a_base() {
            span_lint(
                cx,
                DOC_MARKDOWN,
                span,
                "you should put bare URLs between `<`/`>` or make a proper Markdown link",
            );

            return;
        }
    }

    // We assume that mixed-case words are not meant to be put inside backticks. (Issue #2343)
    if has_underscore(word) && has_hyphen(word) {
        return;
    }

    if has_underscore(word) || word.contains("::") || is_camel_case(word) {
        let mut applicability = Applicability::MachineApplicable;

        span_lint_and_sugg(
            cx,
            DOC_MARKDOWN,
            span,
            "item in documentation is missing backticks",
            "try",
            format!("`{}`", snippet_with_applicability(cx, span, "..", &mut applicability)),
            applicability,
        );
    }
}

struct FindPanicUnwrap<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    panic_span: Option<Span>,
    typeck_results: &'tcx ty::TypeckResults<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for FindPanicUnwrap<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.panic_span.is_some() {
            return;
        }

        // check for `begin_panic`
        if_chain! {
            if let ExprKind::Call(func_expr, _) = expr.kind;
            if let ExprKind::Path(QPath::Resolved(_, path)) = func_expr.kind;
            if let Some(path_def_id) = path.res.opt_def_id();
            if match_panic_def_id(self.cx, path_def_id);
            if is_expn_of(expr.span, "unreachable").is_none();
            if !is_expn_of_debug_assertions(expr.span);
            then {
                self.panic_span = Some(expr.span);
            }
        }

        // check for `assert_eq` or `assert_ne`
        if is_expn_of(expr.span, "assert_eq").is_some() || is_expn_of(expr.span, "assert_ne").is_some() {
            self.panic_span = Some(expr.span);
        }

        // check for `unwrap`
        if let Some(arglists) = method_chain_args(expr, &["unwrap"]) {
            let receiver_ty = self.typeck_results.expr_ty(&arglists[0][0]).peel_refs();
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

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}

fn is_expn_of_debug_assertions(span: Span) -> bool {
    const MACRO_NAMES: &[&str] = &["debug_assert", "debug_assert_eq", "debug_assert_ne"];
    MACRO_NAMES.iter().any(|name| is_expn_of(span, name).is_some())
}
