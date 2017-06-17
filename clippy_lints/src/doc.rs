use itertools::Itertools;
use pulldown_cmark;
use rustc::lint::*;
use syntax::ast;
use syntax::codemap::{Span, BytePos};
use syntax_pos::Pos;
use utils::span_lint;

/// **What it does:** Checks for the presence of `_`, `::` or camel-case words
/// outside ticks in documentation.
///
/// **Why is this bad?** *Rustdoc* supports markdown formatting, `_`, `::` and
/// camel-case probably indicates some code which should be included between
/// ticks. `_` can also be used for empasis in markdown, this lint tries to
/// consider that.
///
/// **Known problems:** Lots of bad docs won’t be fixed, what the lint checks
/// for is limited, and there are still false positives.
///
/// **Examples:**
/// ```rust
/// /// Do something with the foo_bar parameter. See also that::other::module::foo.
/// // ^ `foo_bar` and `that::other::module::foo` should be ticked.
/// fn doit(foo_bar) { .. }
/// ```
declare_lint! {
    pub DOC_MARKDOWN,
    Warn,
    "presence of `_`, `::` or camel-case outside backticks in documentation"
}

#[derive(Clone)]
pub struct Doc {
    valid_idents: Vec<String>,
}

impl Doc {
    pub fn new(valid_idents: Vec<String>) -> Self {
        Doc { valid_idents: valid_idents }
    }
}

impl LintPass for Doc {
    fn get_lints(&self) -> LintArray {
        lint_array![DOC_MARKDOWN]
    }
}

impl EarlyLintPass for Doc {
    fn check_crate(&mut self, cx: &EarlyContext, krate: &ast::Crate) {
        check_attrs(cx, &self.valid_idents, &krate.attrs);
    }

    fn check_item(&mut self, cx: &EarlyContext, item: &ast::Item) {
        check_attrs(cx, &self.valid_idents, &item.attrs);
    }
}

struct Parser<'a> {
    parser: pulldown_cmark::Parser<'a>,
}

impl<'a> Parser<'a> {
    fn new(parser: pulldown_cmark::Parser<'a>) -> Parser<'a> {
        Self { parser }
    }
}

impl<'a> Iterator for Parser<'a> {
    type Item = (usize, pulldown_cmark::Event<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.parser.get_offset();
        self.parser.next().map(|event| (offset, event))
    }
}

/// Cleanup documentation decoration (`///` and such).
///
/// We can't use `syntax::attr::AttributeMethods::with_desugared_doc` or
/// `syntax::parse::lexer::comments::strip_doc_comment_decoration` because we need to keep track of
/// the spans but this function is inspired from the later.
#[allow(cast_possible_truncation)]
pub fn strip_doc_comment_decoration(comment: &str, span: Span) -> (String, Vec<(usize, Span)>) {
    // one-line comments lose their prefix
    const ONELINERS: &'static [&'static str] = &["///!", "///", "//!", "//"];
    for prefix in ONELINERS {
        if comment.starts_with(*prefix) {
            let doc = &comment[prefix.len()..];
            let mut doc = doc.to_owned();
            doc.push('\n');
            return (
                doc.to_owned(),
                vec![(doc.len(), Span { lo: span.lo + BytePos(prefix.len() as u32), ..span })]
            );
        }
    }

    if comment.starts_with("/*") {
        let doc = &comment[3..comment.len() - 2];
        let mut sizes = vec![];

        for line in doc.lines() {
            let offset = line.as_ptr() as usize - comment.as_ptr() as usize;
            debug_assert_eq!(offset as u32 as usize, offset);

            // +1 for the newline
            sizes.push((line.len()+1, Span { lo: span.lo + BytePos(offset as u32), ..span }));
        }

        return (doc.to_string(), sizes);
    }

    panic!("not a doc-comment: {}", comment);
}

pub fn check_attrs<'a>(cx: &EarlyContext, valid_idents: &[String], attrs: &'a [ast::Attribute]) {
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
        } else if let Some(name) = attr.name() {
            // ignore mix of sugared and non-sugared doc
            if name == "doc" {
                return;
            }
        }
    }

    let mut current = 0;
    for &mut (ref mut offset, _) in &mut spans {
        let offset_copy = *offset;
        *offset = current;
        current += offset_copy;
    }

    if !doc.is_empty() {
        let parser = Parser::new(pulldown_cmark::Parser::new(&doc));
        let parser = parser.coalesce(|x, y| {
            use pulldown_cmark::Event::*;

            let x_offset = x.0;
            let y_offset = y.0;

            match (x.1, y.1) {
                (Text(x), Text(y)) => {
                    let mut x = x.into_owned();
                    x.push_str(&y);
                    Ok((x_offset, Text(x.into())))
                }
                (x, y) => Err(((x_offset, x), (y_offset, y))),
            }
        });
        check_doc(cx, valid_idents, parser, &spans);
    }
}

fn check_doc<'a, Events: Iterator<Item=(usize, pulldown_cmark::Event<'a>)>>(
    cx: &EarlyContext,
    valid_idents: &[String],
    docs: Events,
    spans: &[(usize, Span)]
) {
    use pulldown_cmark::Event::*;
    use pulldown_cmark::Tag::*;

    let mut in_code = false;

    for (offset, event) in docs {
        match event {
            Start(CodeBlock(_)) | Start(Code) => in_code = true,
            End(CodeBlock(_)) | End(Code) => in_code = false,
            Start(_tag) | End(_tag) => (), // We don't care about other tags
            Html(_html) | InlineHtml(_html) => (), // HTML is weird, just ignore it
            SoftBreak => (),
            HardBreak => (),
            FootnoteReference(text) | Text(text) => {
                if !in_code {
                    let index = match spans.binary_search_by(|c| c.0.cmp(&offset)) {
                        Ok(o) => o,
                        Err(e) => e-1,
                    };

                    let (begin, span) = spans[index];

                    // Adjust for the begining of the current `Event`
                    let span = Span {
                        lo: span.lo + BytePos::from_usize(offset - begin),
                        ..span
                    };

                    check_text(cx, valid_idents, &text, span);
                }
            },
        }
    }
}

fn check_text(cx: &EarlyContext, valid_idents: &[String], text: &str, span: Span) {
    for word in text.split_whitespace() {
        // Trim punctuation as in `some comment (see foo::bar).`
        //                                                   ^^
        // Or even as in `_foo bar_` which is emphasized.
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());

        if valid_idents.iter().any(|i| i == word) {
            continue;
        }

        // Adjust for the current word
        let offset = word.as_ptr() as usize - text.as_ptr() as usize;
        let span = Span {
            lo: span.lo + BytePos::from_usize(offset),
            hi: span.lo + BytePos::from_usize(offset + word.len()),
            ..span
        };

        check_word(cx, word, span);
    }
}

fn check_word(cx: &EarlyContext, word: &str, span: Span) {
    /// Checks if a string is camel-case, ie. contains at least two uppercase letter (`Clippy` is
    /// ok) and one lower-case letter (`NASA` is ok). Plural are also excluded (`IDs` is ok).
    fn is_camel_case(s: &str) -> bool {
        if s.starts_with(|c: char| c.is_digit(10)) {
            return false;
        }

        let s = if s.ends_with('s') {
            &s[..s.len() - 1]
        } else {
            s
        };

        s.chars().all(char::is_alphanumeric) && s.chars().filter(|&c| c.is_uppercase()).take(2).count() > 1 &&
        s.chars().filter(|&c| c.is_lowercase()).take(1).count() > 0
    }

    fn has_underscore(s: &str) -> bool {
        s != "_" && !s.contains("\\_") && s.contains('_')
    }

    if has_underscore(word) || word.contains("::") || is_camel_case(word) {
        span_lint(cx,
                  DOC_MARKDOWN,
                  span,
                  &format!("you should put `{}` between ticks in the documentation", word));
    }
}
