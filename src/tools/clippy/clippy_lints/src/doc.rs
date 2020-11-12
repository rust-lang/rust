use crate::utils::{implements_trait, is_entrypoint_fn, is_type_diagnostic_item, return_ty, span_lint};
use if_chain::if_chain;
use itertools::Itertools;
use rustc_ast::ast::{Async, AttrKind, Attribute, FnRetTy, ItemKind};
use rustc_ast::token::CommentKind;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_errors::emitter::EmitterWriter;
use rustc_errors::Handler;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_parse::maybe_new_parser_from_source_str;
use rustc_session::parse::ParseSess;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::{BytePos, FilePathMapping, MultiSpan, SourceMap, Span};
use rustc_span::{sym, FileName, Pos};
use std::io;
use std::ops::Range;
use url::Url;

declare_clippy_lint! {
    /// **What it does:** Checks for the presence of `_`, `::` or camel-case words
    /// outside ticks in documentation.
    ///
    /// **Why is this bad?** *Rustdoc* supports markdown formatting, `_`, `::` and
    /// camel-case probably indicates some code which should be included between
    /// ticks. `_` can also be used for emphasis in markdown, this lint tries to
    /// consider that.
    ///
    /// **Known problems:** Lots of bad docs won’t be fixed, what the lint checks
    /// for is limited, and there are still false positives.
    ///
    /// In addition, when writing documentation comments, including `[]` brackets
    /// inside a link text would trip the parser. Therfore, documenting link with
    /// `[`SmallVec<[T; INLINE_CAPACITY]>`]` and then [`SmallVec<[T; INLINE_CAPACITY]>`]: SmallVec
    /// would fail.
    ///
    /// **Examples:**
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
    /// **What it does:** Checks for the doc comments of publicly visible
    /// unsafe functions and warns if there is no `# Safety` section.
    ///
    /// **Why is this bad?** Unsafe functions should document their safety
    /// preconditions, so that users can be sure they are using them safely.
    ///
    /// **Known problems:** None.
    ///
    /// **Examples:**
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
    /// **What it does:** Checks the doc comments of publicly visible functions that
    /// return a `Result` type and warns if there is no `# Errors` section.
    ///
    /// **Why is this bad?** Documenting the type of errors that can be returned from a
    /// function can help callers write code to handle the errors appropriately.
    ///
    /// **Known problems:** None.
    ///
    /// **Examples:**
    ///
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
    /// **What it does:** Checks for `fn main() { .. }` in doctests
    ///
    /// **Why is this bad?** The test can be shorter (and likely more readable)
    /// if the `fn main()` is left implicit.
    ///
    /// **Known problems:** None.
    ///
    /// **Examples:**
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

impl_lint_pass!(DocMarkdown => [DOC_MARKDOWN, MISSING_SAFETY_DOC, MISSING_ERRORS_DOC, NEEDLESS_DOCTEST_MAIN]);

impl<'tcx> LateLintPass<'tcx> for DocMarkdown {
    fn check_crate(&mut self, cx: &LateContext<'tcx>, krate: &'tcx hir::Crate<'_>) {
        check_attrs(cx, &self.valid_idents, &krate.item.attrs);
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        let headers = check_attrs(cx, &self.valid_idents, &item.attrs);
        match item.kind {
            hir::ItemKind::Fn(ref sig, _, body_id) => {
                if !(is_entrypoint_fn(cx, cx.tcx.hir().local_def_id(item.hir_id).to_def_id())
                    || in_external_macro(cx.tcx.sess, item.span))
                {
                    lint_for_missing_headers(cx, item.hir_id, item.span, sig, headers, Some(body_id));
                }
            },
            hir::ItemKind::Impl {
                of_trait: ref trait_ref,
                ..
            } => {
                self.in_trait_impl = trait_ref.is_some();
            },
            _ => {},
        }
    }

    fn check_item_post(&mut self, _cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if let hir::ItemKind::Impl { .. } = item.kind {
            self.in_trait_impl = false;
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
        let headers = check_attrs(cx, &self.valid_idents, &item.attrs);
        if let hir::TraitItemKind::Fn(ref sig, ..) = item.kind {
            if !in_external_macro(cx.tcx.sess, item.span) {
                lint_for_missing_headers(cx, item.hir_id, item.span, sig, headers, None);
            }
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
        let headers = check_attrs(cx, &self.valid_idents, &item.attrs);
        if self.in_trait_impl || in_external_macro(cx.tcx.sess, item.span) {
            return;
        }
        if let hir::ImplItemKind::Fn(ref sig, body_id) = item.kind {
            lint_for_missing_headers(cx, item.hir_id, item.span, sig, headers, Some(body_id));
        }
    }
}

fn lint_for_missing_headers<'tcx>(
    cx: &LateContext<'tcx>,
    hir_id: hir::HirId,
    span: impl Into<MultiSpan> + Copy,
    sig: &hir::FnSig<'_>,
    headers: DocHeaders,
    body_id: Option<hir::BodyId>,
) {
    if !cx.access_levels.is_exported(hir_id) {
        return; // Private functions do not require doc comments
    }
    if !headers.safety && sig.header.unsafety == hir::Unsafety::Unsafe {
        span_lint(
            cx,
            MISSING_SAFETY_DOC,
            span,
            "unsafe function's docs miss `# Safety` section",
        );
    }
    if !headers.errors {
        if is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym::result_type) {
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
                let def_id = cx.tcx.hir().body_owner_def_id(body_id);
                let mir = cx.tcx.optimized_mir(def_id.to_def_id());
                let ret_ty = mir.return_ty();
                if implements_trait(cx, ret_ty, future, &[]);
                if let ty::Opaque(_, subs) = ret_ty.kind();
                if let Some(gen) = subs.types().next();
                if let ty::Generator(_, subs, _) = gen.kind();
                if is_type_diagnostic_item(cx, subs.as_generator().return_ty(), sym::result_type);
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
        while let Some(c) = chars.next() {
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
}

fn check_attrs<'a>(cx: &LateContext<'_>, valid_idents: &FxHashSet<String>, attrs: &'a [Attribute]) -> DocHeaders {
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
        };
    }

    let parser = pulldown_cmark::Parser::new(&doc).into_offset_iter();
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

const RUST_CODE: &[&str] = &["rust", "no_run", "should_panic", "compile_fail", "edition2018"];

fn check_doc<'a, Events: Iterator<Item = (pulldown_cmark::Event<'a>, Range<usize>)>>(
    cx: &LateContext<'_>,
    valid_idents: &FxHashSet<String>,
    events: Events,
    spans: &[(usize, Span)],
) -> DocHeaders {
    // true if a safety header was found
    use pulldown_cmark::CodeBlockKind;
    use pulldown_cmark::Event::{
        Code, End, FootnoteReference, HardBreak, Html, Rule, SoftBreak, Start, TaskListMarker, Text,
    };
    use pulldown_cmark::Tag::{CodeBlock, Heading, Link};

    let mut headers = DocHeaders {
        safety: false,
        errors: false,
    };
    let mut in_code = false;
    let mut in_link = None;
    let mut in_heading = false;
    let mut is_rust = false;
    for (event, range) in events {
        match event {
            Start(CodeBlock(ref kind)) => {
                in_code = true;
                if let CodeBlockKind::Fenced(lang) = kind {
                    is_rust =
                        lang.is_empty() || !lang.contains("ignore") && lang.split(',').any(|i| RUST_CODE.contains(&i));
                }
            },
            End(CodeBlock(_)) => {
                in_code = false;
                is_rust = false;
            },
            Start(Link(_, url, _)) => in_link = Some(url),
            End(Link(..)) => in_link = None,
            Start(Heading(_)) => in_heading = true,
            End(Heading(_)) => in_heading = false,
            Start(_tag) | End(_tag) => (), // We don't care about other tags
            Html(_html) => (),             // HTML is weird, just ignore it
            SoftBreak | HardBreak | TaskListMarker(_) | Code(_) | Rule => (),
            FootnoteReference(text) | Text(text) => {
                if Some(&text) == in_link.as_ref() {
                    // Probably a link of the form `<http://example.com>`
                    // Which are represented as a link to "http://example.com" with
                    // text "http://example.com" by pulldown-cmark
                    continue;
                }
                headers.safety |= in_heading && text.trim() == "Safety";
                headers.errors |= in_heading && text.trim() == "Errors";
                let index = match spans.binary_search_by(|c| c.0.cmp(&range.start)) {
                    Ok(o) => o,
                    Err(e) => e - 1,
                };
                let (begin, span) = spans[index];
                if in_code {
                    if is_rust {
                        check_code(cx, &text, span);
                    }
                } else {
                    // Adjust for the beginning of the current `Event`
                    let span = span.with_lo(span.lo() + BytePos::from_usize(range.start - begin));

                    check_text(cx, valid_idents, &text, span);
                }
            },
        }
    }
    headers
}

fn check_code(cx: &LateContext<'_>, text: &str, span: Span) {
    fn has_needless_main(code: &str) -> bool {
        let filename = FileName::anon_source_code(code);

        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let emitter = EmitterWriter::new(box io::sink(), None, false, false, false, None, false);
        let handler = Handler::with_emitter(false, None, box emitter);
        let sess = ParseSess::with_span_handler(handler, sm);

        let mut parser = match maybe_new_parser_from_source_str(&sess, filename, code.into()) {
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
            match parser.parse_item() {
                Ok(Some(item)) => match &item.kind {
                    // Tests with one of these items are ignored
                    ItemKind::Static(..)
                    | ItemKind::Const(..)
                    | ItemKind::ExternCrate(..)
                    | ItemKind::ForeignMod(..) => return false,
                    // We found a main function ...
                    ItemKind::Fn(_, sig, _, Some(block)) if item.ident.name == sym::main => {
                        let is_async = matches!(sig.header.asyncness, Async::Yes{..});
                        let returns_nothing = match &sig.decl.output {
                            FnRetTy::Default(..) => true,
                            FnRetTy::Ty(ty) if ty.kind.is_unit() => true,
                            _ => false,
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
    }

    if has_needless_main(text) {
        span_lint(cx, NEEDLESS_DOCTEST_MAIN, span, "needless `fn main` in doctest");
    }
}

fn check_text(cx: &LateContext<'_>, valid_idents: &FxHashSet<String>, text: &str, span: Span) {
    for word in text.split(|c: char| c.is_whitespace() || c == '\'') {
        // Trim punctuation as in `some comment (see foo::bar).`
        //                                                   ^^
        // Or even as in `_foo bar_` which is emphasized.
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());

        if valid_idents.contains(word) {
            continue;
        }

        // Adjust for the current word
        let offset = word.as_ptr() as usize - text.as_ptr() as usize;
        let span = Span::new(
            span.lo() + BytePos::from_usize(offset),
            span.lo() + BytePos::from_usize(offset + word.len()),
            span.ctxt(),
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

    // We assume that mixed-case words are not meant to be put inside bacticks. (Issue #2343)
    if has_underscore(word) && has_hyphen(word) {
        return;
    }

    if has_underscore(word) || word.contains("::") || is_camel_case(word) {
        span_lint(
            cx,
            DOC_MARKDOWN,
            span,
            &format!("you should put `{}` between ticks in the documentation", word),
        );
    }
}
