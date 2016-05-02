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
                        check_doc(cx, valid_idents, real_doc, span);
                    }
                }
            }
        }
    }
}

macro_rules! jump_to {
    // Get the next character’s first byte UTF-8 friendlyly.
    (@next_char, $chars: expr, $len: expr) => {{
        if let Some(&(pos, _)) = $chars.peek() {
            pos
        } else {
            $len
        }
    }};

    // Jump to the next `$c`. If no such character is found, give up.
    ($chars: expr, $c: expr, $len: expr) => {{
        if $chars.find(|&(_, c)| c == $c).is_some() {
            jump_to!(@next_char, $chars, $len)
        }
        else {
            return;
        }
    }};
}

#[allow(while_let_loop)] // #362
pub fn check_doc(cx: &EarlyContext, valid_idents: &[String], doc: &str, span: Span) {
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

    /// Character that can appear in a word
    fn is_word_char(c: char) -> bool {
        match c {
            t if t.is_alphanumeric() => true,
            ':' | '_' => true,
            _ => false,
        }
    }

    #[allow(cast_possible_truncation)]
    fn word_span(mut span: Span, begin: usize, end: usize) -> Span {
        debug_assert_eq!(end as u32 as usize, end);
        debug_assert_eq!(begin as u32 as usize, begin);
        span.hi = span.lo + BytePos(end as u32);
        span.lo = span.lo + BytePos(begin as u32);
        span
    }

    let len = doc.len();
    let mut chars = doc.char_indices().peekable();
    let mut current_word_begin = 0;
    loop {
        match chars.next() {
            Some((_, c)) => {
                match c {
                    c if c.is_whitespace() => {
                        current_word_begin = jump_to!(@next_char, chars, len);
                    }
                    '`' => {
                        current_word_begin = jump_to!(chars, '`', len);
                    }
                    '[' => {
                        let end = jump_to!(chars, ']', len);
                        let link_text = &doc[current_word_begin + 1..end];
                        let word_span = word_span(span, current_word_begin + 1, end + 1);

                        match chars.peek() {
                            Some(&(_, c)) => {
                                // Trying to parse a link. Let’s ignore the link.

                                // FIXME: how does markdown handles such link?
                                // https://en.wikipedia.org/w/index.php?title=)
                                match c {
                                    '(' => { // inline link
                                        current_word_begin = jump_to!(chars, ')', len);
                                        check_doc(cx, valid_idents, link_text, word_span);
                                    }
                                    '[' => { // reference link
                                        current_word_begin = jump_to!(chars, ']', len);
                                        check_doc(cx, valid_idents, link_text, word_span);
                                    }
                                    ':' => { // reference link
                                        current_word_begin = jump_to!(chars, '\n', len);
                                    }
                                    _ => { // automatic reference link
                                        current_word_begin = jump_to!(@next_char, chars, len);
                                        check_doc(cx, valid_idents, link_text, word_span);
                                    }
                                }
                            }
                            None => return,
                        }
                    }
                    _ => {
                        let end = match chars.find(|&(_, c)| !is_word_char(c)) {
                            Some((end, _)) => end,
                            None => len,
                        };
                        let word_span = word_span(span, current_word_begin, end);
                        check_word(cx, valid_idents, &doc[current_word_begin..end], word_span);
                        current_word_begin = jump_to!(@next_char, chars, len);
                    }
                }
            }
            None => break,
        }
    }
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
