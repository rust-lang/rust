//! There are many AstNodes, but only a few tokens, so we hand-write them here.

use std::ops::Range;
use std::{borrow::Cow, num::ParseIntError};

use rustc_literal_escaper::{
    EscapeError, MixedUnit, unescape_byte, unescape_byte_str, unescape_c_str, unescape_char,
    unescape_str,
};
use stdx::always;

use crate::{
    TextRange, TextSize,
    ast::{self, AstToken},
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
        text.find('\n').is_some_and(|idx| text[idx + 1..].contains('\n'))
    }
}

#[derive(Debug)]
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
    const RAW_PREFIX: &'static str;
    fn unescape(s: &str, callback: impl FnMut(Range<usize>, Result<char, EscapeError>));
    fn is_raw(&self) -> bool {
        self.text().starts_with(Self::RAW_PREFIX)
    }
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
    fn text_without_quotes(&self) -> &str {
        let text = self.text();
        let Some(offsets) = self.text_range_between_quotes() else { return text };
        &text[offsets - self.syntax().text_range().start()]
    }
    fn open_quote_text_range(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.quotes.0)
    }
    fn close_quote_text_range(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.quotes.1)
    }
    fn escaped_char_ranges(&self, cb: &mut dyn FnMut(TextRange, Result<char, EscapeError>)) {
        let Some(text_range_no_quotes) = self.text_range_between_quotes() else { return };

        let start = self.syntax().text_range().start();
        let text = &self.text()[text_range_no_quotes - start];
        let offset = text_range_no_quotes.start() - start;

        Self::unescape(text, &mut |range: Range<usize>, unescaped_char| {
            if let Some((s, e)) = range.start.try_into().ok().zip(range.end.try_into().ok()) {
                cb(TextRange::new(s, e) + offset, unescaped_char);
            }
        });
    }
    fn map_range_up(&self, range: TextRange) -> Option<TextRange> {
        let contents_range = self.text_range_between_quotes()?;
        if always!(TextRange::up_to(contents_range.len()).contains_range(range)) {
            Some(range + contents_range.start())
        } else {
            None
        }
    }
}

impl IsString for ast::String {
    const RAW_PREFIX: &'static str = "r";
    fn unescape(s: &str, cb: impl FnMut(Range<usize>, Result<char, EscapeError>)) {
        unescape_str(s, cb)
    }
}

impl ast::String {
    pub fn value(&self) -> Result<Cow<'_, str>, EscapeError> {
        let text = self.text();
        let text_range = self.text_range_between_quotes().ok_or(EscapeError::LoneSlash)?;
        let text = &text[text_range - self.syntax().text_range().start()];
        if self.is_raw() {
            return Ok(Cow::Borrowed(text));
        }

        let mut buf = String::new();
        let mut prev_end = 0;
        let mut has_error = None;
        unescape_str(text, |char_range, unescaped_char| {
            match (unescaped_char, buf.capacity() == 0) {
                (Ok(c), false) => buf.push(c),
                (Ok(_), true) if char_range.len() == 1 && char_range.start == prev_end => {
                    prev_end = char_range.end
                }
                (Ok(c), true) => {
                    buf.reserve_exact(text.len());
                    buf.push_str(&text[..prev_end]);
                    buf.push(c);
                }
                (Err(e), _) => has_error = Some(e),
            }
        });

        match (has_error, buf.capacity() == 0) {
            (Some(e), _) => Err(e),
            (None, true) => Ok(Cow::Borrowed(text)),
            (None, false) => Ok(Cow::Owned(buf)),
        }
    }
}

impl IsString for ast::ByteString {
    const RAW_PREFIX: &'static str = "br";
    fn unescape(s: &str, mut callback: impl FnMut(Range<usize>, Result<char, EscapeError>)) {
        unescape_byte_str(s, |range, res| callback(range, res.map(char::from)))
    }
}

impl ast::ByteString {
    pub fn value(&self) -> Result<Cow<'_, [u8]>, EscapeError> {
        let text = self.text();
        let text_range = self.text_range_between_quotes().ok_or(EscapeError::LoneSlash)?;
        let text = &text[text_range - self.syntax().text_range().start()];
        if self.is_raw() {
            return Ok(Cow::Borrowed(text.as_bytes()));
        }

        let mut buf: Vec<u8> = Vec::new();
        let mut prev_end = 0;
        let mut has_error = None;
        unescape_byte_str(text, |char_range, unescaped_byte| {
            match (unescaped_byte, buf.capacity() == 0) {
                (Ok(b), false) => buf.push(b),
                (Ok(_), true) if char_range.len() == 1 && char_range.start == prev_end => {
                    prev_end = char_range.end
                }
                (Ok(b), true) => {
                    buf.reserve_exact(text.len());
                    buf.extend_from_slice(&text.as_bytes()[..prev_end]);
                    buf.push(b);
                }
                (Err(e), _) => has_error = Some(e),
            }
        });

        match (has_error, buf.capacity() == 0) {
            (Some(e), _) => Err(e),
            (None, true) => Ok(Cow::Borrowed(text.as_bytes())),
            (None, false) => Ok(Cow::Owned(buf)),
        }
    }
}

impl IsString for ast::CString {
    const RAW_PREFIX: &'static str = "cr";
    // NOTE: This method should only be used for highlighting ranges. The unescaped
    // char/byte is not used. For simplicity, we return an arbitrary placeholder char.
    fn unescape(s: &str, mut callback: impl FnMut(Range<usize>, Result<char, EscapeError>)) {
        unescape_c_str(s, |range, _res| callback(range, Ok('_')))
    }
}

impl ast::CString {
    pub fn value(&self) -> Result<Cow<'_, [u8]>, EscapeError> {
        let text = self.text();
        let text_range = self.text_range_between_quotes().ok_or(EscapeError::LoneSlash)?;
        let text = &text[text_range - self.syntax().text_range().start()];
        if self.is_raw() {
            return Ok(Cow::Borrowed(text.as_bytes()));
        }

        let mut buf = Vec::new();
        let mut prev_end = 0;
        let mut has_error = None;
        let extend_unit = |buf: &mut Vec<u8>, unit: MixedUnit| match unit {
            MixedUnit::Char(c) => buf.extend(c.encode_utf8(&mut [0; 4]).as_bytes()),
            MixedUnit::HighByte(b) => buf.push(b),
        };
        unescape_c_str(text, |char_range, unescaped| match (unescaped, buf.capacity() == 0) {
            (Ok(u), false) => extend_unit(&mut buf, u),
            (Ok(_), true) if char_range.len() == 1 && char_range.start == prev_end => {
                prev_end = char_range.end
            }
            (Ok(u), true) => {
                buf.reserve_exact(text.len());
                buf.extend(&text.as_bytes()[..prev_end]);
                extend_unit(&mut buf, u);
            }
            (Err(e), _) => has_error = Some(e),
        });

        match (has_error, buf.capacity() == 0) {
            (Some(e), _) => Err(e),
            (None, true) => Ok(Cow::Borrowed(text.as_bytes())),
            (None, false) => Ok(Cow::Owned(buf)),
        }
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

    pub fn value(&self) -> Result<u128, ParseIntError> {
        let (_, text, _) = self.split_into_parts();
        u128::from_str_radix(&text.replace('_', ""), self.radix() as u32)
    }

    pub fn suffix(&self) -> Option<&str> {
        let (_, _, suffix) = self.split_into_parts();
        if suffix.is_empty() { None } else { Some(suffix) }
    }

    pub fn value_string(&self) -> String {
        let (_, text, _) = self.split_into_parts();
        text.replace('_', "")
    }
}

impl ast::FloatNumber {
    pub fn split_into_parts(&self) -> (&str, &str) {
        let text = self.text();
        let mut float_text = self.text();
        let mut suffix = "";
        let mut indices = text.char_indices();
        if let Some((mut suffix_start, c)) = indices.by_ref().find(|(_, c)| c.is_ascii_alphabetic())
        {
            if c == 'e' || c == 'E' {
                if let Some(suffix_start_tuple) = indices.find(|(_, c)| c.is_ascii_alphabetic()) {
                    suffix_start = suffix_start_tuple.0;

                    float_text = &text[..suffix_start];
                    suffix = &text[suffix_start..];
                }
            } else {
                float_text = &text[..suffix_start];
                suffix = &text[suffix_start..];
            }
        }

        (float_text, suffix)
    }

    pub fn suffix(&self) -> Option<&str> {
        let (_, suffix) = self.split_into_parts();
        if suffix.is_empty() { None } else { Some(suffix) }
    }

    pub fn value_string(&self) -> String {
        let (text, _) = self.split_into_parts();
        text.replace('_', "")
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

impl ast::Char {
    pub fn value(&self) -> Result<char, EscapeError> {
        let mut text = self.text();
        if text.starts_with('\'') {
            text = &text[1..];
        } else {
            return Err(EscapeError::ZeroChars);
        }
        if text.ends_with('\'') {
            text = &text[0..text.len() - 1];
        }

        unescape_char(text)
    }
}

impl ast::Byte {
    pub fn value(&self) -> Result<u8, EscapeError> {
        let mut text = self.text();
        if text.starts_with("b\'") {
            text = &text[2..];
        } else {
            return Err(EscapeError::ZeroChars);
        }
        if text.ends_with('\'') {
            text = &text[0..text.len() - 1];
        }

        unescape_byte(text)
    }
}

#[cfg(test)]
mod tests {
    use rustc_apfloat::ieee::Quad as f128;

    use crate::ast::{self, FloatNumber, IntNumber, make};

    fn check_float_suffix<'a>(lit: &str, expected: impl Into<Option<&'a str>>) {
        assert_eq!(FloatNumber { syntax: make::tokens::literal(lit) }.suffix(), expected.into());
    }

    fn check_int_suffix<'a>(lit: &str, expected: impl Into<Option<&'a str>>) {
        assert_eq!(IntNumber { syntax: make::tokens::literal(lit) }.suffix(), expected.into());
    }

    // FIXME(#17451) Use `expected: f128` once `f128` is stabilised.
    fn check_float_value(lit: &str, expected: &str) {
        let expected = Some(expected.parse::<f128>().unwrap());
        assert_eq!(
            FloatNumber { syntax: make::tokens::literal(lit) }.value_string().parse::<f128>().ok(),
            expected
        );
        assert_eq!(
            IntNumber { syntax: make::tokens::literal(lit) }.value_string().parse::<f128>().ok(),
            expected
        );
    }

    fn check_int_value(lit: &str, expected: impl Into<Option<u128>>) {
        assert_eq!(IntNumber { syntax: make::tokens::literal(lit) }.value().ok(), expected.into());
    }

    #[test]
    fn test_float_number_suffix() {
        check_float_suffix("123.0", None);
        check_float_suffix("123f32", "f32");
        check_float_suffix("123.0e", None);
        check_float_suffix("123.0e4", None);
        check_float_suffix("123.0ef16", "f16");
        check_float_suffix("123.0E4f32", "f32");
        check_float_suffix("1_2_3.0_f128", "f128");
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
            ast::String { syntax: make::tokens::literal(&format!("\"{lit}\"")) }
                .value()
                .as_deref()
                .ok(),
            expected.into()
        );
    }

    #[test]
    fn test_string_escape() {
        check_string_value(r"foobar", "foobar");
        check_string_value(r"\foobar", None);
        check_string_value(r"\nfoobar", "\nfoobar");
        check_string_value(r"C:\\Windows\\System32\\", "C:\\Windows\\System32\\");
        check_string_value(r"\x61bcde", "abcde");
        check_string_value(
            r"a\
bcde", "abcde",
        );
    }

    fn check_byte_string_value<'a, const N: usize>(
        lit: &str,
        expected: impl Into<Option<&'a [u8; N]>>,
    ) {
        assert_eq!(
            ast::ByteString { syntax: make::tokens::literal(&format!("b\"{lit}\"")) }
                .value()
                .as_deref()
                .ok(),
            expected.into().map(|value| &value[..])
        );
    }

    #[test]
    fn test_byte_string_escape() {
        check_byte_string_value(r"foobar", b"foobar");
        check_byte_string_value(r"\foobar", None::<&[u8; 0]>);
        check_byte_string_value(r"\nfoobar", b"\nfoobar");
        check_byte_string_value(r"C:\\Windows\\System32\\", b"C:\\Windows\\System32\\");
        check_byte_string_value(r"\x61bcde", b"abcde");
        check_byte_string_value(
            r"a\
bcde", b"abcde",
        );
    }

    #[test]
    fn test_value_underscores() {
        check_float_value("1.3_4665449586950493453___6_f128", "1.346654495869504934536");
        check_float_value("1.234567891011121_f64", "1.234567891011121");
        check_float_value("1__0.__0__f32", "10.0");
        check_float_value("3._0_f16", "3.0");
        check_int_value("0b__1_0_", 2);
        check_int_value("1_1_1_1_1_1", 111111);
    }
}
