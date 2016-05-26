use rustc::lint::*;
use syntax::ast;
use syntax::codemap::{Span, BytePos};
use utils::span_lint;

/// **What it does:** This lint checks for the presence of `_`, `::` or camel-case words outside
/// ticks in documentation.
///
/// **Why is this bad?** *Rustdoc* supports markdown formatting, `_`, `::` and camel-case probably
/// indicates some code which should be included between ticks. `_` can also be used for empasis in
/// markdown, this lint tries to consider that.
///
/// **Known problems:** Lots of bad docs won’t be fixed, what the lint checks for is limited.
///
/// **Examples:**
/// ```rust
/// /// Do something with the foo_bar parameter. See also that::other::module::foo.
/// // ^ `foo_bar` and `that::other::module::foo` should be ticked.
/// fn doit(foo_bar) { .. }
/// ```
declare_lint! {
    pub DOC_MARKDOWN, Warn,
    "checks for the presence of `_`, `::` or camel-case outside ticks in documentation"
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

pub fn check_attrs<'a>(cx: &EarlyContext, valid_idents: &[String], attrs: &'a [ast::Attribute]) {
    let mut docs = vec![];

    let mut in_multiline = false;
    for attr in attrs {
        if attr.node.is_sugared_doc {
            if let ast::MetaItemKind::NameValue(_, ref doc) = attr.node.value.node {
                if let ast::LitKind::Str(ref doc, _) = doc.node {
                    // doc comments start with `///` or `//!`
                    let real_doc = &doc[3..];
                    let mut span = attr.span;
                    span.lo = span.lo + BytePos(3);

                    // check for multiline code blocks
                    if real_doc.trim_left().starts_with("```") {
                        in_multiline = !in_multiline;
                    }
                    if !in_multiline {
                        docs.push((real_doc, span));
                    }
                }
            }
        }
    }

    for (doc, span) in docs {
        let _ = check_doc(cx, valid_idents, doc, span);
    }
}

#[allow(while_let_loop)] // #362
pub fn check_doc(cx: &EarlyContext, valid_idents: &[String], doc: &str, span: Span) -> Result<(), ()> {
    // In markdown, `_` can be used to emphasize something, or, is a raw `_` depending on context.
    // There really is no markdown specification that would disambiguate this properly. This is
    // what GitHub and Rustdoc do:
    //
    // foo_bar test_quz    → foo_bar test_quz
    // foo_bar_baz         → foo_bar_baz (note that the “official” spec says this should be emphasized)
    // _foo bar_ test_quz_ → <em>foo bar</em> test_quz_
    // \_foo bar\_         → _foo bar_
    // (_baz_)             → (<em>baz</em>)
    // foo _ bar _ baz     → foo _ bar _ baz

    /// Character that can appear in a path
    fn is_path_char(c: char) -> bool {
        match c {
            t if t.is_alphanumeric() => true,
            ':' | '_' => true,
            _ => false,
        }
    }

    #[derive(Clone, Debug)]
    struct Parser<'a> {
        link: bool,
        line: &'a str,
        span: Span,
        current_word_begin: usize,
        new_line: bool,
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn advance_begin(&mut self) {
            self.current_word_begin = self.pos;
        }

        fn peek(&self) -> Option<char> {
            self.line[self.pos..].chars().next()
        }

        fn jump_to(&mut self, n: char) -> Result<(), ()> {
            while let Some(c) = self.next() {
                if c == n {
                    self.advance_begin();
                    return Ok(());
                }
            }

            return Err(());
        }

        fn put_back(&mut self, c: char) {
            self.pos -= c.len_utf8();
        }

        #[allow(cast_possible_truncation)]
        fn word(&self) -> (&'a str, Span) {
            let begin = self.current_word_begin;
            let end = self.pos;

            debug_assert_eq!(end as u32 as usize, end);
            debug_assert_eq!(begin as u32 as usize, begin);

            let mut span = self.span;
            span.hi = span.lo + BytePos(end as u32);
            span.lo = span.lo + BytePos(begin as u32);

            (&self.line[begin..end], span)
        }
    }

    impl<'a> Iterator for Parser<'a> {
        type Item = char;

        fn next(&mut self) -> Option<char> {
            let mut chars = self.line[self.pos..].chars();
            let c = chars.next();

            if let Some(c) = c {
                self.pos += c.len_utf8();
            } else {
                // TODO: new line
            }

            c
        }
    }

    let mut parser = Parser {
        link: false,
        line: doc,
        span: span,
        current_word_begin: 0,
        new_line: true,
        pos: 0,
    };

    loop {
        match parser.next() {
            Some(c) => {
                match c {
                    '#' if new_line => { // don’t warn on titles
                        try!(parser.jump_to('\n'));
                    }
                    '`' => {
                        try!(parser.jump_to('`'));
                    }
                    '[' => {
                        // Check for a reference definition `[foo]:` at the beginning of a line
                        let mut link = true;
                        if parser.new_line {
                            let mut lookup_parser = parser.clone();
                            if let Some(_) = lookup_parser.find(|&c| c == ']') {
                                if let Some(':') = lookup_parser.next() {
                                    try!(lookup_parser.jump_to(')'));
                                    parser = lookup_parser;
                                    link = false;
                                }
                            }
                        }

                        parser.advance_begin();
                        parser.link = link;
                    }
                    ']' if parser.link => {
                        parser.link = false;

                        match parser.peek() {
                            Some('(') => try!(parser.jump_to(')')),
                            Some('[') => try!(parser.jump_to(']')),
                            Some(_) => continue,
                            None => return Err(()),
                        }
                    }
                    c if !is_path_char(c) => {
                        parser.advance_begin();
                    }
                    _ => {
                        if let Some(c) = parser.find(|&c| !is_path_char(c)) {
                            parser.put_back(c);
                        }

                        let (word, span) = parser.word();
                        check_word(cx, valid_idents, word, span);
                        parser.advance_begin();
                    }
                }

                parser.new_line = c == '\n' || (parser.new_line && c.is_whitespace());
            }
            None => break,
        }
    }

    Ok(())
}

fn check_word(cx: &EarlyContext, valid_idents: &[String], word: &str, span: Span) {
    /// Checks if a string a camel-case, ie. contains at least two uppercase letter (`Clippy` is
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

        s.chars().all(char::is_alphanumeric) &&
        s.chars().filter(|&c| c.is_uppercase()).take(2).count() > 1 &&
        s.chars().filter(|&c| c.is_lowercase()).take(1).count() > 0
    }

    fn has_underscore(s: &str) -> bool {
        s != "_" && !s.contains("\\_") && s.contains('_')
    }

    // Trim punctuation as in `some comment (see foo::bar).`
    //                                                   ^^
    // Or even as in `_foo bar_` which is emphasized.
    let word = word.trim_matches(|c: char| !c.is_alphanumeric());

    if valid_idents.iter().any(|i| i == word) {
        return;
    }

    if has_underscore(word) || word.contains("::") || is_camel_case(word) {
        span_lint(cx,
                  DOC_MARKDOWN,
                  span,
                  &format!("you should put `{}` between ticks in the documentation", word));
    }
}
