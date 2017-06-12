use rustc::lint::*;
use syntax::ast;
use syntax::codemap::{Span, BytePos};
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

/// Cleanup documentation decoration (`///` and such).
///
/// We can't use `syntax::attr::AttributeMethods::with_desugared_doc` or
/// `syntax::parse::lexer::comments::strip_doc_comment_decoration` because we need to keep track of
/// the span but this function is inspired from the later.
#[allow(cast_possible_truncation)]
pub fn strip_doc_comment_decoration((comment, span): (String, Span)) -> Vec<(String, Span)> {
    // one-line comments lose their prefix
    const ONELINERS: &'static [&'static str] = &["///!", "///", "//!", "//"];
    for prefix in ONELINERS {
        if comment.starts_with(*prefix) {
            return vec![(comment[prefix.len()..].to_owned(),
                         Span { lo: span.lo + BytePos(prefix.len() as u32), ..span })];
        }
    }

    if comment.starts_with("/*") {
        return comment[3..comment.len() - 2]
            .lines()
            .map(|line| {
                let offset = line.as_ptr() as usize - comment.as_ptr() as usize;
                debug_assert_eq!(offset as u32 as usize, offset);

                (line.to_owned(), Span { lo: span.lo + BytePos(offset as u32), ..span })
            })
            .collect();
    }

    panic!("not a doc-comment: {}", comment);
}

pub fn check_attrs<'a>(cx: &EarlyContext, valid_idents: &[String], attrs: &'a [ast::Attribute]) {
    let mut docs = vec![];

    for attr in attrs {
        if attr.is_sugared_doc {
            if let Some(ref doc) = attr.value_str() {
                let doc = doc.to_string();
                docs.extend_from_slice(&strip_doc_comment_decoration((doc, attr.span)));
            }
        }
    }

    if !docs.is_empty() {
        let _ = check_doc(cx, valid_idents, &docs);
    }
}

#[allow(while_let_loop)] // #362
fn check_doc(cx: &EarlyContext, valid_idents: &[String], docs: &[(String, Span)]) -> Result<(), ()> {
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
    /// This type is used to iterate through the documentation characters, keeping the span at the
    /// same time.
    struct Parser<'a> {
        /// First byte of the current potential match
        current_word_begin: usize,
        /// List of lines and their associated span
        docs: &'a [(String, Span)],
        /// Index of the current line we are parsing
        line: usize,
        /// Whether we are in a link
        link: bool,
        /// Whether we are at the beginning of a line
        new_line: bool,
        /// Whether we were to the end of a line last time `next` was called
        reset: bool,
        /// The position of the current character within the current line
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn advance_begin(&mut self) {
            self.current_word_begin = self.pos;
        }

        fn line(&self) -> (&'a str, Span) {
            let (ref doc, span) = self.docs[self.line];
            (doc, span)
        }

        fn peek(&self) -> Option<char> {
            self.line().0[self.pos..].chars().next()
        }

        #[allow(while_let_on_iterator)] // borrowck complains about for
        fn jump_to(&mut self, n: char) -> Result<bool, ()> {
            while let Some((new_line, c)) = self.next() {
                if c == n {
                    self.advance_begin();
                    return Ok(new_line);
                }
            }

            Err(())
        }

        fn next_line(&mut self) {
            self.pos = 0;
            self.current_word_begin = 0;
            self.line += 1;
            self.new_line = true;
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

            let (doc, mut span) = self.line();
            span.hi = span.lo + BytePos(end as u32);
            span.lo = span.lo + BytePos(begin as u32);

            (&doc[begin..end], span)
        }
    }

    impl<'a> Iterator for Parser<'a> {
        type Item = (bool, char);

        fn next(&mut self) -> Option<(bool, char)> {
            if self.line < self.docs.len() {
                if self.reset {
                    self.line += 1;
                    self.reset = false;
                    self.pos = 0;
                    self.current_word_begin = 0;
                }

                let mut chars = self.line().0[self.pos..].chars();
                let c = chars.next();

                if let Some(c) = c {
                    self.pos += c.len_utf8();
                    let new_line = self.new_line;
                    self.new_line = c == '\n' || (self.new_line && c.is_whitespace());
                    Some((new_line, c))
                } else if self.line == self.docs.len() - 1 {
                    None
                } else {
                    self.new_line = true;
                    self.reset = true;
                    self.pos += 1;
                    Some((true, '\n'))
                }
            } else {
                None
            }
        }
    }

    let mut parser = Parser {
        current_word_begin: 0,
        docs: docs,
        line: 0,
        link: false,
        new_line: true,
        reset: false,
        pos: 0,
    };

    /// Check for fanced code block.
    macro_rules! check_block {
        ($parser:expr, $c:tt, $new_line:expr) => {{
            check_block!($parser, $c, $c, $new_line)
        }};

        ($parser:expr, $c:pat, $c_expr:expr, $new_line:expr) => {{
            fn check_block(parser: &mut Parser, new_line: bool) -> Result<bool, ()> {
                if new_line {
                    let mut lookup_parser = parser.clone();
                    if let (Some((false, $c)), Some((false, $c))) = (lookup_parser.next(), lookup_parser.next()) {
                        *parser = lookup_parser;
                        // 3 or more ` or ~ open a code block to be closed with the same number of ` or ~
                        let mut open_count = 3;
                        while let Some((false, $c)) = parser.next() {
                            open_count += 1;
                        }

                        loop {
                            loop {
                                if try!(parser.jump_to($c_expr)) {
                                    break;
                                }
                            }

                            lookup_parser = parser.clone();
                            let a = lookup_parser.next();
                            let b = lookup_parser.next();
                            if let (Some((false, $c)), Some((false, $c))) = (a, b) {
                                let mut close_count = 3;
                                while let Some((false, $c)) = lookup_parser.next() {
                                    close_count += 1;
                                }

                                if close_count == open_count {
                                    *parser = lookup_parser;
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }

                Ok(false)
            }

            check_block(&mut $parser, $new_line)
        }};
    }

    loop {
        match parser.next() {
            Some((new_line, c)) => {
                match c {
                    '#' if new_line => {
                        // don’t warn on titles
                        parser.next_line();
                    },
                    '`' => {
                        if try!(check_block!(parser, '`', new_line)) {
                            continue;
                        }

                        // not a code block, just inline code
                        try!(parser.jump_to('`'));
                    },
                    '~' => {
                        if try!(check_block!(parser, '~', new_line)) {
                            continue;
                        }

                        // ~ does not introduce inline code, but two of them introduce
                        // strikethrough. Too bad for the consistency but we don't care about
                        // strikethrough.
                    },
                    '[' => {
                        // Check for a reference definition `[foo]:` at the beginning of a line
                        let mut link = true;

                        if new_line {
                            let mut lookup_parser = parser.clone();
                            if lookup_parser.any(|(_, c)| c == ']') {
                                if let Some((_, ':')) = lookup_parser.next() {
                                    lookup_parser.next_line();
                                    parser = lookup_parser;
                                    link = false;
                                }
                            }
                        }

                        parser.advance_begin();
                        parser.link = link;
                    },
                    ']' if parser.link => {
                        parser.link = false;

                        match parser.peek() {
                            Some('(') => {
                                try!(parser.jump_to(')'));
                            },
                            Some('[') => {
                                try!(parser.jump_to(']'));
                            },
                            Some(_) => continue,
                            None => return Err(()),
                        }
                    },
                    c if !is_path_char(c) => {
                        parser.advance_begin();
                    },
                    _ => {
                        if let Some((_, c)) = parser.find(|&(_, c)| !is_path_char(c)) {
                            parser.put_back(c);
                        }

                        let (word, span) = parser.word();
                        check_word(cx, valid_idents, word, span);
                        parser.advance_begin();
                    },
                }

            },
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

        s.chars().all(char::is_alphanumeric) && s.chars().filter(|&c| c.is_uppercase()).take(2).count() > 1 &&
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
