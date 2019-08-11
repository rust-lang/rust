use crate::utils::span_lint;
use itertools::Itertools;
use pulldown_cmark;
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, impl_lint_pass};
use rustc_data_structures::fx::FxHashSet;
use std::ops::Range;
use syntax::ast;
use syntax::source_map::{BytePos, Span};
use syntax_pos::Pos;
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
    /// **Examples:**
    /// ```rust
    /// /// Do something with the foo_bar parameter. See also
    /// /// that::other::module::foo.
    /// // ^ `foo_bar` and `that::other::module::foo` should be ticked.
    /// fn doit(foo_bar: usize) {}
    /// ```
    pub DOC_MARKDOWN,
    pedantic,
    "presence of `_`, `::` or camel-case outside backticks in documentation"
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct DocMarkdown {
    valid_idents: FxHashSet<String>,
}

impl DocMarkdown {
    pub fn new(valid_idents: FxHashSet<String>) -> Self {
        Self { valid_idents }
    }
}

impl_lint_pass!(DocMarkdown => [DOC_MARKDOWN]);

impl EarlyLintPass for DocMarkdown {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &ast::Crate) {
        check_attrs(cx, &self.valid_idents, &krate.attrs);
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        check_attrs(cx, &self.valid_idents, &item.attrs);
    }
}

/// Cleanup documentation decoration (`///` and such).
///
/// We can't use `syntax::attr::AttributeMethods::with_desugared_doc` or
/// `syntax::parse::lexer::comments::strip_doc_comment_decoration` because we
/// need to keep track of
/// the spans but this function is inspired from the later.
#[allow(clippy::cast_possible_truncation)]
pub fn strip_doc_comment_decoration(comment: &str, span: Span) -> (String, Vec<(usize, Span)>) {
    // one-line comments lose their prefix
    const ONELINERS: &[&str] = &["///!", "///", "//!", "//"];
    for prefix in ONELINERS {
        if comment.starts_with(*prefix) {
            let doc = &comment[prefix.len()..];
            let mut doc = doc.to_owned();
            doc.push('\n');
            return (
                doc.to_owned(),
                vec![(doc.len(), span.with_lo(span.lo() + BytePos(prefix.len() as u32)))],
            );
        }
    }

    if comment.starts_with("/*") {
        let doc = &comment[3..comment.len() - 2];
        let mut sizes = vec![];
        let mut contains_initial_stars = false;
        for line in doc.lines() {
            let offset = line.as_ptr() as usize - comment.as_ptr() as usize;
            debug_assert_eq!(offset as u32 as usize, offset);
            contains_initial_stars |= line.trim_start().starts_with('*');
            // +1 for the newline
            sizes.push((line.len() + 1, span.with_lo(span.lo() + BytePos(offset as u32))));
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
        return (no_stars, sizes);
    }

    panic!("not a doc-comment: {}", comment);
}

pub fn check_attrs<'a>(cx: &EarlyContext<'_>, valid_idents: &FxHashSet<String>, attrs: &'a [ast::Attribute]) {
    let mut doc = String::new();
    let mut spans = vec![];

    for attr in attrs {
        if attr.is_sugared_doc {
            if let Some(ref current) = attr.value_str() {
                let current = current.to_string();
                let (current, current_spans) = strip_doc_comment_decoration(&current, attr.span);
                spans.extend_from_slice(&current_spans);
                doc.push_str(&current);
            }
        } else if attr.check_name(sym!(doc)) {
            // ignore mix of sugared and non-sugared doc
            return;
        }
    }

    let mut current = 0;
    for &mut (ref mut offset, _) in &mut spans {
        let offset_copy = *offset;
        *offset = current;
        current += offset_copy;
    }

    if !doc.is_empty() {
        let parser = pulldown_cmark::Parser::new(&doc).into_offset_iter();
        // Iterate over all `Events` and combine consecutive events into one
        let events = parser.coalesce(|previous, current| {
            use pulldown_cmark::Event::*;

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
        check_doc(cx, valid_idents, events, &spans);
    }
}

fn check_doc<'a, Events: Iterator<Item = (pulldown_cmark::Event<'a>, Range<usize>)>>(
    cx: &EarlyContext<'_>,
    valid_idents: &FxHashSet<String>,
    events: Events,
    spans: &[(usize, Span)],
) {
    use pulldown_cmark::Event::*;
    use pulldown_cmark::Tag::*;

    let mut in_code = false;
    let mut in_link = None;

    for (event, range) in events {
        match event {
            Start(CodeBlock(_)) => in_code = true,
            End(CodeBlock(_)) => in_code = false,
            Start(Link(_, url, _)) => in_link = Some(url),
            End(Link(..)) => in_link = None,
            Start(_tag) | End(_tag) => (),         // We don't care about other tags
            Html(_html) | InlineHtml(_html) => (), // HTML is weird, just ignore it
            SoftBreak | HardBreak | TaskListMarker(_) | Code(_) => (),
            FootnoteReference(text) | Text(text) => {
                if Some(&text) == in_link.as_ref() {
                    // Probably a link of the form `<http://example.com>`
                    // Which are represented as a link to "http://example.com" with
                    // text "http://example.com" by pulldown-cmark
                    continue;
                }

                if !in_code {
                    let index = match spans.binary_search_by(|c| c.0.cmp(&range.start)) {
                        Ok(o) => o,
                        Err(e) => e - 1,
                    };

                    let (begin, span) = spans[index];

                    // Adjust for the beginning of the current `Event`
                    let span = span.with_lo(span.lo() + BytePos::from_usize(range.start - begin));

                    check_text(cx, valid_idents, &text, span);
                }
            },
        }
    }
}

fn check_text(cx: &EarlyContext<'_>, valid_idents: &FxHashSet<String>, text: &str, span: Span) {
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

fn check_word(cx: &EarlyContext<'_>, word: &str, span: Span) {
    /// Checks if a string is camel-case, i.e., contains at least two uppercase
    /// letters (`Clippy` is ok) and one lower-case letter (`NASA` is ok).
    /// Plurals are also excluded (`IDs` is ok).
    fn is_camel_case(s: &str) -> bool {
        if s.starts_with(|c: char| c.is_digit(10)) {
            return false;
        }

        let s = if s.ends_with('s') { &s[..s.len() - 1] } else { s };

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
