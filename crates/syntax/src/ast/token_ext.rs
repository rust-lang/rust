//! There are many AstNodes, but only a few tokens, so we hand-write them here.

use std::borrow::Cow;

use rustc_lexer::unescape::{unescape_literal, Mode};

use crate::{
    ast::{self, AstToken},
    TextRange, TextSize,
};

impl ast::Comment {
    pub fn kind(&self) -> CommentKind {
        CommentKind::from_text(self.text())
    }

    pub fn is_doc(&self) -> bool {
        self.kind().doc.is_some()
    }

    pub fn is_inner(&self) -> bool {
        self.kind().doc == Some(CommentPlacement::Inner)
    }

    pub fn is_outer(&self) -> bool {
        self.kind().doc == Some(CommentPlacement::Outer)
    }

    pub fn prefix(&self) -> &'static str {
        let &(prefix, _kind) = CommentKind::BY_PREFIX
            .iter()
            .find(|&(prefix, kind)| self.kind() == *kind && self.text().starts_with(prefix))
            .unwrap();
        prefix
    }

    /// Returns the textual content of a doc comment node as a single string with prefix and suffix
    /// removed.
    pub fn doc_comment(&self) -> Option<&str> {
        let kind = self.kind();
        match kind {
            CommentKind { shape, doc: Some(_) } => {
                let prefix = kind.prefix();
                let text = &self.text()[prefix.len()..];
                let text = if shape == CommentShape::Block {
                    text.strip_suffix("*/").unwrap_or(text)
                } else {
                    text
                };
                Some(text)
            }
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct CommentKind {
    pub shape: CommentShape,
    pub doc: Option<CommentPlacement>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CommentShape {
    Line,
    Block,
}

impl CommentShape {
    pub fn is_line(self) -> bool {
        self == CommentShape::Line
    }

    pub fn is_block(self) -> bool {
        self == CommentShape::Block
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CommentPlacement {
    Inner,
    Outer,
}

impl CommentKind {
    const BY_PREFIX: [(&'static str, CommentKind); 9] = [
        ("/**/", CommentKind { shape: CommentShape::Block, doc: None }),
        ("/***", CommentKind { shape: CommentShape::Block, doc: None }),
        ("////", CommentKind { shape: CommentShape::Line, doc: None }),
        ("///", CommentKind { shape: CommentShape::Line, doc: Some(CommentPlacement::Outer) }),
        ("//!", CommentKind { shape: CommentShape::Line, doc: Some(CommentPlacement::Inner) }),
        ("/**", CommentKind { shape: CommentShape::Block, doc: Some(CommentPlacement::Outer) }),
        ("/*!", CommentKind { shape: CommentShape::Block, doc: Some(CommentPlacement::Inner) }),
        ("//", CommentKind { shape: CommentShape::Line, doc: None }),
        ("/*", CommentKind { shape: CommentShape::Block, doc: None }),
    ];

    pub(crate) fn from_text(text: &str) -> CommentKind {
        let &(_prefix, kind) = CommentKind::BY_PREFIX
            .iter()
            .find(|&(prefix, _kind)| text.starts_with(prefix))
            .unwrap();
        kind
    }

    pub fn prefix(&self) -> &'static str {
        let &(prefix, _) =
            CommentKind::BY_PREFIX.iter().rev().find(|(_, kind)| kind == self).unwrap();
        prefix
    }
}

impl ast::Whitespace {
    pub fn spans_multiple_lines(&self) -> bool {
        let text = self.text();
        text.find('\n').map_or(false, |idx| text[idx + 1..].contains('\n'))
    }
}

pub struct QuoteOffsets {
    pub quotes: (TextRange, TextRange),
    pub contents: TextRange,
}

impl QuoteOffsets {
    fn new(literal: &str) -> Option<QuoteOffsets> {
        let left_quote = literal.find('"')?;
        let right_quote = literal.rfind('"')?;
        if left_quote == right_quote {
            // `literal` only contains one quote
            return None;
        }

        let start = TextSize::from(0);
        let left_quote = TextSize::try_from(left_quote).unwrap() + TextSize::of('"');
        let right_quote = TextSize::try_from(right_quote).unwrap();
        let end = TextSize::of(literal);

        let res = QuoteOffsets {
            quotes: (TextRange::new(start, left_quote), TextRange::new(right_quote, end)),
            contents: TextRange::new(left_quote, right_quote),
        };
        Some(res)
    }
}

pub trait IsString: AstToken {
    fn quote_offsets(&self) -> Option<QuoteOffsets> {
        let text = self.text();
        let offsets = QuoteOffsets::new(text)?;
        let o = self.syntax().text_range().start();
        let offsets = QuoteOffsets {
            quotes: (offsets.quotes.0 + o, offsets.quotes.1 + o),
            contents: offsets.contents + o,
        };
        Some(offsets)
    }
    fn text_range_between_quotes(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.contents)
    }
    fn open_quote_text_range(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.quotes.0)
    }
    fn close_quote_text_range(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.quotes.1)
    }
}

impl IsString for ast::String {}

impl ast::String {
    pub fn is_raw(&self) -> bool {
        self.text().starts_with('r')
    }
    pub fn map_range_up(&self, range: TextRange) -> Option<TextRange> {
        let contents_range = self.text_range_between_quotes()?;
        assert!(TextRange::up_to(contents_range.len()).contains_range(range));
        Some(range + contents_range.start())
    }

    pub fn value(&self) -> Option<Cow<'_, str>> {
        if self.is_raw() {
            let text = self.text();
            let text =
                &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];
            return Some(Cow::Borrowed(text));
        }

        let text = self.text();
        let text = &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];

        let mut buf = String::new();
        let mut text_iter = text.chars();
        let mut has_error = false;
        unescape_literal(text, Mode::Str, &mut |char_range, unescaped_char| match (
            unescaped_char,
            buf.capacity() == 0,
        ) {
            (Ok(c), false) => buf.push(c),
            (Ok(c), true) if char_range.len() == 1 && Some(c) == text_iter.next() => (),
            (Ok(c), true) => {
                buf.reserve_exact(text.len());
                buf.push_str(&text[..char_range.start]);
                buf.push(c);
            }
            (Err(_), _) => has_error = true,
        });

        match (has_error, buf.capacity() == 0) {
            (true, _) => None,
            (false, true) => Some(Cow::Borrowed(text)),
            (false, false) => Some(Cow::Owned(buf)),
        }
    }
}

impl IsString for ast::ByteString {}

impl ast::ByteString {
    pub fn is_raw(&self) -> bool {
        self.text().starts_with("br")
    }

    pub fn value(&self) -> Option<Cow<'_, [u8]>> {
        if self.is_raw() {
            let text = self.text();
            let text =
                &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];
            return Some(Cow::Borrowed(text.as_bytes()));
        }

        let text = self.text();
        let text = &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];

        let mut buf: Vec<u8> = Vec::new();
        let mut text_iter = text.chars();
        let mut has_error = false;
        unescape_literal(text, Mode::ByteStr, &mut |char_range, unescaped_char| match (
            unescaped_char,
            buf.capacity() == 0,
        ) {
            (Ok(c), false) => buf.push(c as u8),
            (Ok(c), true) if char_range.len() == 1 && Some(c) == text_iter.next() => (),
            (Ok(c), true) => {
                buf.reserve_exact(text.len());
                buf.extend_from_slice(text[..char_range.start].as_bytes());
                buf.push(c as u8);
            }
            (Err(_), _) => has_error = true,
        });

        match (has_error, buf.capacity() == 0) {
            (true, _) => None,
            (false, true) => Some(Cow::Borrowed(text.as_bytes())),
            (false, false) => Some(Cow::Owned(buf)),
        }
    }
}

#[derive(Debug)]
pub enum FormatSpecifier {
    Open,
    Close,
    Integer,
    Identifier,
    Colon,
    Fill,
    Align,
    Sign,
    NumberSign,
    Zero,
    DollarSign,
    Dot,
    Asterisk,
    QuestionMark,
}

pub trait HasFormatSpecifier: AstToken {
    fn char_ranges(
        &self,
    ) -> Option<Vec<(TextRange, Result<char, rustc_lexer::unescape::EscapeError>)>>;

    fn lex_format_specifier<F>(&self, mut callback: F)
    where
        F: FnMut(TextRange, FormatSpecifier),
    {
        let char_ranges = match self.char_ranges() {
            Some(char_ranges) => char_ranges,
            None => return,
        };
        let mut chars = char_ranges.iter().peekable();

        while let Some((range, first_char)) = chars.next() {
            match first_char {
                Ok('{') => {
                    // Format specifier, see syntax at https://doc.rust-lang.org/std/fmt/index.html#syntax
                    if let Some((_, Ok('{'))) = chars.peek() {
                        // Escaped format specifier, `{{`
                        chars.next();
                        continue;
                    }

                    callback(*range, FormatSpecifier::Open);

                    // check for integer/identifier
                    match chars
                        .peek()
                        .and_then(|next| next.1.as_ref().ok())
                        .copied()
                        .unwrap_or_default()
                    {
                        '0'..='9' => {
                            // integer
                            read_integer(&mut chars, &mut callback);
                        }
                        c if c == '_' || c.is_alphabetic() => {
                            // identifier
                            read_identifier(&mut chars, &mut callback);
                        }
                        _ => {}
                    }

                    if let Some((_, Ok(':'))) = chars.peek() {
                        skip_char_and_emit(&mut chars, FormatSpecifier::Colon, &mut callback);

                        // check for fill/align
                        let mut cloned = chars.clone().take(2);
                        let first = cloned
                            .next()
                            .and_then(|next| next.1.as_ref().ok())
                            .copied()
                            .unwrap_or_default();
                        let second = cloned
                            .next()
                            .and_then(|next| next.1.as_ref().ok())
                            .copied()
                            .unwrap_or_default();
                        match second {
                            '<' | '^' | '>' => {
                                // alignment specifier, first char specifies fillment
                                skip_char_and_emit(
                                    &mut chars,
                                    FormatSpecifier::Fill,
                                    &mut callback,
                                );
                                skip_char_and_emit(
                                    &mut chars,
                                    FormatSpecifier::Align,
                                    &mut callback,
                                );
                            }
                            _ => match first {
                                '<' | '^' | '>' => {
                                    skip_char_and_emit(
                                        &mut chars,
                                        FormatSpecifier::Align,
                                        &mut callback,
                                    );
                                }
                                _ => {}
                            },
                        }

                        // check for sign
                        match chars
                            .peek()
                            .and_then(|next| next.1.as_ref().ok())
                            .copied()
                            .unwrap_or_default()
                        {
                            '+' | '-' => {
                                skip_char_and_emit(
                                    &mut chars,
                                    FormatSpecifier::Sign,
                                    &mut callback,
                                );
                            }
                            _ => {}
                        }

                        // check for `#`
                        if let Some((_, Ok('#'))) = chars.peek() {
                            skip_char_and_emit(
                                &mut chars,
                                FormatSpecifier::NumberSign,
                                &mut callback,
                            );
                        }

                        // check for `0`
                        let mut cloned = chars.clone().take(2);
                        let first = cloned.next().and_then(|next| next.1.as_ref().ok()).copied();
                        let second = cloned.next().and_then(|next| next.1.as_ref().ok()).copied();

                        if first == Some('0') && second != Some('$') {
                            skip_char_and_emit(&mut chars, FormatSpecifier::Zero, &mut callback);
                        }

                        // width
                        match chars
                            .peek()
                            .and_then(|next| next.1.as_ref().ok())
                            .copied()
                            .unwrap_or_default()
                        {
                            '0'..='9' => {
                                read_integer(&mut chars, &mut callback);
                                if let Some((_, Ok('$'))) = chars.peek() {
                                    skip_char_and_emit(
                                        &mut chars,
                                        FormatSpecifier::DollarSign,
                                        &mut callback,
                                    );
                                }
                            }
                            c if c == '_' || c.is_alphabetic() => {
                                read_identifier(&mut chars, &mut callback);

                                if chars.peek().and_then(|next| next.1.as_ref().ok()).copied()
                                    == Some('?')
                                {
                                    skip_char_and_emit(
                                        &mut chars,
                                        FormatSpecifier::QuestionMark,
                                        &mut callback,
                                    );
                                }

                                // can be either width (indicated by dollar sign, or type in which case
                                // the next sign has to be `}`)
                                let next =
                                    chars.peek().and_then(|next| next.1.as_ref().ok()).copied();

                                match next {
                                    Some('$') => skip_char_and_emit(
                                        &mut chars,
                                        FormatSpecifier::DollarSign,
                                        &mut callback,
                                    ),
                                    Some('}') => {
                                        skip_char_and_emit(
                                            &mut chars,
                                            FormatSpecifier::Close,
                                            &mut callback,
                                        );
                                        continue;
                                    }
                                    _ => continue,
                                };
                            }
                            _ => {}
                        }

                        // precision
                        if let Some((_, Ok('.'))) = chars.peek() {
                            skip_char_and_emit(&mut chars, FormatSpecifier::Dot, &mut callback);

                            match chars
                                .peek()
                                .and_then(|next| next.1.as_ref().ok())
                                .copied()
                                .unwrap_or_default()
                            {
                                '*' => {
                                    skip_char_and_emit(
                                        &mut chars,
                                        FormatSpecifier::Asterisk,
                                        &mut callback,
                                    );
                                }
                                '0'..='9' => {
                                    read_integer(&mut chars, &mut callback);
                                    if let Some((_, Ok('$'))) = chars.peek() {
                                        skip_char_and_emit(
                                            &mut chars,
                                            FormatSpecifier::DollarSign,
                                            &mut callback,
                                        );
                                    }
                                }
                                c if c == '_' || c.is_alphabetic() => {
                                    read_identifier(&mut chars, &mut callback);
                                    if chars.peek().and_then(|next| next.1.as_ref().ok()).copied()
                                        != Some('$')
                                    {
                                        continue;
                                    }
                                    skip_char_and_emit(
                                        &mut chars,
                                        FormatSpecifier::DollarSign,
                                        &mut callback,
                                    );
                                }
                                _ => {
                                    continue;
                                }
                            }
                        }

                        // type
                        match chars
                            .peek()
                            .and_then(|next| next.1.as_ref().ok())
                            .copied()
                            .unwrap_or_default()
                        {
                            '?' => {
                                skip_char_and_emit(
                                    &mut chars,
                                    FormatSpecifier::QuestionMark,
                                    &mut callback,
                                );
                            }
                            c if c == '_' || c.is_alphabetic() => {
                                read_identifier(&mut chars, &mut callback);

                                if chars.peek().and_then(|next| next.1.as_ref().ok()).copied()
                                    == Some('?')
                                {
                                    skip_char_and_emit(
                                        &mut chars,
                                        FormatSpecifier::QuestionMark,
                                        &mut callback,
                                    );
                                }
                            }
                            _ => {}
                        }
                    }

                    match chars.peek() {
                        Some((_, Ok('}'))) => {
                            skip_char_and_emit(&mut chars, FormatSpecifier::Close, &mut callback);
                        }
                        Some((_, _)) | None => continue,
                    }
                }
                _ => {
                    while let Some((_, Ok(next_char))) = chars.peek() {
                        if next_char == &'{' {
                            break;
                        }
                        chars.next();
                    }
                }
            };
        }

        fn skip_char_and_emit<'a, I, F>(
            chars: &mut std::iter::Peekable<I>,
            emit: FormatSpecifier,
            callback: &mut F,
        ) where
            I: Iterator<Item = &'a (TextRange, Result<char, rustc_lexer::unescape::EscapeError>)>,
            F: FnMut(TextRange, FormatSpecifier),
        {
            let (range, _) = chars.next().unwrap();
            callback(*range, emit);
        }

        fn read_integer<'a, I, F>(chars: &mut std::iter::Peekable<I>, callback: &mut F)
        where
            I: Iterator<Item = &'a (TextRange, Result<char, rustc_lexer::unescape::EscapeError>)>,
            F: FnMut(TextRange, FormatSpecifier),
        {
            let (mut range, c) = chars.next().unwrap();
            assert!(c.as_ref().unwrap().is_ascii_digit());
            while let Some((r, Ok(next_char))) = chars.peek() {
                if next_char.is_ascii_digit() {
                    chars.next();
                    range = range.cover(*r);
                } else {
                    break;
                }
            }
            callback(range, FormatSpecifier::Integer);
        }

        fn read_identifier<'a, I, F>(chars: &mut std::iter::Peekable<I>, callback: &mut F)
        where
            I: Iterator<Item = &'a (TextRange, Result<char, rustc_lexer::unescape::EscapeError>)>,
            F: FnMut(TextRange, FormatSpecifier),
        {
            let (mut range, c) = chars.next().unwrap();
            assert!(c.as_ref().unwrap().is_alphabetic() || *c.as_ref().unwrap() == '_');
            while let Some((r, Ok(next_char))) = chars.peek() {
                if *next_char == '_' || next_char.is_ascii_digit() || next_char.is_alphabetic() {
                    chars.next();
                    range = range.cover(*r);
                } else {
                    break;
                }
            }
            callback(range, FormatSpecifier::Identifier);
        }
    }
}

impl HasFormatSpecifier for ast::String {
    fn char_ranges(
        &self,
    ) -> Option<Vec<(TextRange, Result<char, rustc_lexer::unescape::EscapeError>)>> {
        let text = self.text();
        let text = &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];
        let offset = self.text_range_between_quotes()?.start() - self.syntax().text_range().start();

        let mut res = Vec::with_capacity(text.len());
        unescape_literal(text, Mode::Str, &mut |range, unescaped_char| {
            res.push((
                TextRange::new(range.start.try_into().unwrap(), range.end.try_into().unwrap())
                    + offset,
                unescaped_char,
            ));
        });

        Some(res)
    }
}

impl ast::IntNumber {
    pub fn radix(&self) -> Radix {
        match self.text().get(..2).unwrap_or_default() {
            "0b" => Radix::Binary,
            "0o" => Radix::Octal,
            "0x" => Radix::Hexadecimal,
            _ => Radix::Decimal,
        }
    }

    pub fn split_into_parts(&self) -> (&str, &str, &str) {
        let radix = self.radix();
        let (prefix, mut text) = self.text().split_at(radix.prefix_len());

        let is_suffix_start: fn(&(usize, char)) -> bool = match radix {
            Radix::Hexadecimal => |(_, c)| matches!(c, 'g'..='z' | 'G'..='Z'),
            _ => |(_, c)| c.is_ascii_alphabetic(),
        };

        let mut suffix = "";
        if let Some((suffix_start, _)) = text.char_indices().find(is_suffix_start) {
            let (text2, suffix2) = text.split_at(suffix_start);
            text = text2;
            suffix = suffix2;
        };

        (prefix, text, suffix)
    }

    pub fn value(&self) -> Option<u128> {
        let (_, text, _) = self.split_into_parts();
        let value = u128::from_str_radix(&text.replace("_", ""), self.radix() as u32).ok()?;
        Some(value)
    }

    pub fn suffix(&self) -> Option<&str> {
        let (_, _, suffix) = self.split_into_parts();
        if suffix.is_empty() {
            None
        } else {
            Some(suffix)
        }
    }
}

impl ast::FloatNumber {
    pub fn suffix(&self) -> Option<&str> {
        let text = self.text();
        let mut indices = text.char_indices();
        let (mut suffix_start, c) = indices.by_ref().find(|(_, c)| c.is_ascii_alphabetic())?;
        if c == 'e' || c == 'E' {
            suffix_start = indices.find(|(_, c)| c.is_ascii_alphabetic())?.0;
        }
        Some(&text[suffix_start..])
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Radix {
    Binary = 2,
    Octal = 8,
    Decimal = 10,
    Hexadecimal = 16,
}

impl Radix {
    pub const ALL: &'static [Radix] =
        &[Radix::Binary, Radix::Octal, Radix::Decimal, Radix::Hexadecimal];

    const fn prefix_len(self) -> usize {
        match self {
            Self::Decimal => 0,
            _ => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{self, make, FloatNumber, IntNumber};

    fn check_float_suffix<'a>(lit: &str, expected: impl Into<Option<&'a str>>) {
        assert_eq!(FloatNumber { syntax: make::tokens::literal(lit) }.suffix(), expected.into());
    }

    fn check_int_suffix<'a>(lit: &str, expected: impl Into<Option<&'a str>>) {
        assert_eq!(IntNumber { syntax: make::tokens::literal(lit) }.suffix(), expected.into());
    }

    #[test]
    fn test_float_number_suffix() {
        check_float_suffix("123.0", None);
        check_float_suffix("123f32", "f32");
        check_float_suffix("123.0e", None);
        check_float_suffix("123.0e4", None);
        check_float_suffix("123.0ef32", "f32");
        check_float_suffix("123.0E4f32", "f32");
        check_float_suffix("1_2_3.0_f32", "f32");
    }

    #[test]
    fn test_int_number_suffix() {
        check_int_suffix("123", None);
        check_int_suffix("123i32", "i32");
        check_int_suffix("1_0_1_l_o_l", "l_o_l");
        check_int_suffix("0b11", None);
        check_int_suffix("0o11", None);
        check_int_suffix("0xff", None);
        check_int_suffix("0b11u32", "u32");
        check_int_suffix("0o11u32", "u32");
        check_int_suffix("0xffu32", "u32");
    }

    fn check_string_value<'a>(lit: &str, expected: impl Into<Option<&'a str>>) {
        assert_eq!(
            ast::String { syntax: make::tokens::literal(&format!("\"{}\"", lit)) }
                .value()
                .as_deref(),
            expected.into()
        );
    }

    #[test]
    fn test_string_escape() {
        check_string_value(r"foobar", "foobar");
        check_string_value(r"\foobar", None);
        check_string_value(r"\nfoobar", "\nfoobar");
        check_string_value(r"C:\\Windows\\System32\\", "C:\\Windows\\System32\\");
    }
}
