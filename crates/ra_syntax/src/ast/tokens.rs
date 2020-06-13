//! There are many AstNodes, but only a few tokens, so we hand-write them here.

use std::convert::{TryFrom, TryInto};

use crate::{
    ast::{AstToken, Comment, RawString, String, Whitespace},
    TextRange, TextSize,
};
use rustc_lexer::unescape::{unescape_literal, Mode};

impl Comment {
    pub fn kind(&self) -> CommentKind {
        kind_by_prefix(self.text())
    }

    pub fn prefix(&self) -> &'static str {
        for (prefix, k) in COMMENT_PREFIX_TO_KIND.iter() {
            if *k == self.kind() && self.text().starts_with(prefix) {
                return prefix;
            }
        }
        unreachable!()
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

const COMMENT_PREFIX_TO_KIND: &[(&str, CommentKind)] = {
    use {CommentPlacement::*, CommentShape::*};
    &[
        ("////", CommentKind { shape: Line, doc: None }),
        ("///", CommentKind { shape: Line, doc: Some(Outer) }),
        ("//!", CommentKind { shape: Line, doc: Some(Inner) }),
        ("/**", CommentKind { shape: Block, doc: Some(Outer) }),
        ("/*!", CommentKind { shape: Block, doc: Some(Inner) }),
        ("//", CommentKind { shape: Line, doc: None }),
        ("/*", CommentKind { shape: Block, doc: None }),
    ]
};

fn kind_by_prefix(text: &str) -> CommentKind {
    if text == "/**/" {
        return CommentKind { shape: CommentShape::Block, doc: None };
    }
    for (prefix, kind) in COMMENT_PREFIX_TO_KIND.iter() {
        if text.starts_with(prefix) {
            return *kind;
        }
    }
    panic!("bad comment text: {:?}", text)
}

impl Whitespace {
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

pub trait HasQuotes: AstToken {
    fn quote_offsets(&self) -> Option<QuoteOffsets> {
        let text = self.text().as_str();
        let offsets = QuoteOffsets::new(text)?;
        let o = self.syntax().text_range().start();
        let offsets = QuoteOffsets {
            quotes: (offsets.quotes.0 + o, offsets.quotes.1 + o),
            contents: offsets.contents + o,
        };
        Some(offsets)
    }
    fn open_quote_text_range(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.quotes.0)
    }

    fn close_quote_text_range(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.quotes.1)
    }

    fn text_range_between_quotes(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.contents)
    }
}

impl HasQuotes for String {}
impl HasQuotes for RawString {}

pub trait HasStringValue: HasQuotes {
    fn value(&self) -> Option<std::string::String>;
}

impl HasStringValue for String {
    fn value(&self) -> Option<std::string::String> {
        let text = self.text().as_str();
        let text = &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];

        let mut buf = std::string::String::with_capacity(text.len());
        let mut has_error = false;
        unescape_literal(text, Mode::Str, &mut |_, unescaped_char| match unescaped_char {
            Ok(c) => buf.push(c),
            Err(_) => has_error = true,
        });

        if has_error {
            return None;
        }
        Some(buf)
    }
}

impl HasStringValue for RawString {
    fn value(&self) -> Option<std::string::String> {
        let text = self.text().as_str();
        let text = &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];
        Some(text.to_string())
    }
}

impl RawString {
    pub fn map_range_up(&self, range: TextRange) -> Option<TextRange> {
        let contents_range = self.text_range_between_quotes()?;
        assert!(TextRange::up_to(contents_range.len()).contains_range(range));
        Some(range + contents_range.start())
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
        let char_ranges = if let Some(char_ranges) = self.char_ranges() {
            char_ranges
        } else {
            return;
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
                            }
                            _ => {}
                        }
                    }

                    if let Some((_, Ok('}'))) = chars.peek() {
                        skip_char_and_emit(&mut chars, FormatSpecifier::Close, &mut callback);
                    } else {
                        continue;
                    }
                }
                _ => {
                    while let Some((_, Ok(next_char))) = chars.peek() {
                        match next_char {
                            '{' => break,
                            _ => {}
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

impl HasFormatSpecifier for String {
    fn char_ranges(
        &self,
    ) -> Option<Vec<(TextRange, Result<char, rustc_lexer::unescape::EscapeError>)>> {
        let text = self.text().as_str();
        let text = &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];
        let offset = self.text_range_between_quotes()?.start() - self.syntax().text_range().start();

        let mut res = Vec::with_capacity(text.len());
        unescape_literal(text, Mode::Str, &mut |range, unescaped_char| {
            res.push((
                TextRange::new(range.start.try_into().unwrap(), range.end.try_into().unwrap())
                    + offset,
                unescaped_char,
            ))
        });

        Some(res)
    }
}

impl HasFormatSpecifier for RawString {
    fn char_ranges(
        &self,
    ) -> Option<Vec<(TextRange, Result<char, rustc_lexer::unescape::EscapeError>)>> {
        let text = self.text().as_str();
        let text = &text[self.text_range_between_quotes()? - self.syntax().text_range().start()];
        let offset = self.text_range_between_quotes()?.start() - self.syntax().text_range().start();

        let mut res = Vec::with_capacity(text.len());
        for (idx, c) in text.char_indices() {
            res.push((TextRange::at(idx.try_into().unwrap(), TextSize::of(c)) + offset, Ok(c)));
        }
        Some(res)
    }
}
