//! There are many AstNodes, but only a few tokens, so we hand-write them here.

use crate::{
    ast::{AstToken, Comment, RawString, String, Whitespace},
    TextRange, TextUnit,
};

impl Comment {
    pub fn kind(&self) -> CommentKind {
        kind_by_prefix(self.text())
    }

    pub fn prefix(&self) -> &'static str {
        prefix_by_kind(self.kind())
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
        ("///", CommentKind { shape: Line, doc: Some(Outer) }),
        ("//!", CommentKind { shape: Line, doc: Some(Inner) }),
        ("/**", CommentKind { shape: Block, doc: Some(Outer) }),
        ("/*!", CommentKind { shape: Block, doc: Some(Inner) }),
        ("//", CommentKind { shape: Line, doc: None }),
        ("/*", CommentKind { shape: Block, doc: None }),
    ]
};

fn kind_by_prefix(text: &str) -> CommentKind {
    for (prefix, kind) in COMMENT_PREFIX_TO_KIND.iter() {
        if text.starts_with(prefix) {
            return *kind;
        }
    }
    panic!("bad comment text: {:?}", text)
}

fn prefix_by_kind(kind: CommentKind) -> &'static str {
    for (prefix, k) in COMMENT_PREFIX_TO_KIND.iter() {
        if *k == kind {
            return prefix;
        }
    }
    unreachable!()
}

impl Whitespace {
    pub fn spans_multiple_lines(&self) -> bool {
        let text = self.text();
        text.find('\n').map_or(false, |idx| text[idx + 1..].contains('\n'))
    }
}

pub struct QuoteOffsets {
    pub quotes: [TextRange; 2],
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

        let start = TextUnit::from(0);
        let left_quote = TextUnit::from_usize(left_quote) + TextUnit::of_char('"');
        let right_quote = TextUnit::from_usize(right_quote);
        let end = TextUnit::of_str(literal);

        let res = QuoteOffsets {
            quotes: [TextRange::from_to(start, left_quote), TextRange::from_to(right_quote, end)],
            contents: TextRange::from_to(left_quote, right_quote),
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
            quotes: [offsets.quotes[0] + o, offsets.quotes[1] + o],
            contents: offsets.contents + o,
        };
        Some(offsets)
    }
    fn open_quote_text_range(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.quotes[0])
    }

    fn close_quote_text_range(&self) -> Option<TextRange> {
        self.quote_offsets().map(|it| it.quotes[1])
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
        rustc_lexer::unescape::unescape_str(text, &mut |_, unescaped_char| match unescaped_char {
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
        assert!(range.is_subrange(&TextRange::offset_len(0.into(), contents_range.len())));
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
    fn lex_format_specifier<F>(&self, callback: &mut F)
    where
        F: FnMut(TextRange, FormatSpecifier),
    {
        let src = self.text().as_str();
        let initial_len = src.len();
        let mut chars = src.chars();

        while let Some(first_char) = chars.next() {
            match first_char {
                '{' => {
                    // Format specifier, see syntax at https://doc.rust-lang.org/std/fmt/index.html#syntax
                    if chars.clone().next() == Some('{') {
                        // Escaped format specifier, `{{`
                        chars.next();
                        continue;
                    }

                    let start = initial_len - chars.as_str().len() - first_char.len_utf8();
                    let end = initial_len - chars.as_str().len();
                    callback(
                        TextRange::from_to(TextUnit::from_usize(start), TextUnit::from_usize(end)),
                        FormatSpecifier::Open,
                    );

                    let next_char = if let Some(c) = chars.clone().next() {
                        c
                    } else {
                        break;
                    };

                    // check for integer/identifier
                    match next_char {
                        '0'..='9' => {
                            // integer
                            read_integer(&mut chars, initial_len, callback);
                        }
                        'a'..='z' | 'A'..='Z' | '_' => {
                            // identifier
                            read_identifier(&mut chars, initial_len, callback);
                        }
                        _ => {}
                    }

                    if chars.clone().next() == Some(':') {
                        skip_char_and_emit(
                            &mut chars,
                            initial_len,
                            FormatSpecifier::Colon,
                            callback,
                        );

                        // check for fill/align
                        let mut cloned = chars.clone().take(2);
                        let first = cloned.next().unwrap_or_default();
                        let second = cloned.next().unwrap_or_default();
                        match second {
                            '<' | '^' | '>' => {
                                // alignment specifier, first char specifies fillment
                                skip_char_and_emit(
                                    &mut chars,
                                    initial_len,
                                    FormatSpecifier::Fill,
                                    callback,
                                );
                                skip_char_and_emit(
                                    &mut chars,
                                    initial_len,
                                    FormatSpecifier::Align,
                                    callback,
                                );
                            }
                            _ => match first {
                                '<' | '^' | '>' => {
                                    skip_char_and_emit(
                                        &mut chars,
                                        initial_len,
                                        FormatSpecifier::Align,
                                        callback,
                                    );
                                }
                                _ => {}
                            },
                        }

                        // check for sign
                        match chars.clone().next().unwrap_or_default() {
                            '+' | '-' => {
                                skip_char_and_emit(
                                    &mut chars,
                                    initial_len,
                                    FormatSpecifier::Sign,
                                    callback,
                                );
                            }
                            _ => {}
                        }

                        // check for `#`
                        if let Some('#') = chars.clone().next() {
                            skip_char_and_emit(
                                &mut chars,
                                initial_len,
                                FormatSpecifier::NumberSign,
                                callback,
                            );
                        }

                        // check for `0`
                        let mut cloned = chars.clone().take(2);
                        let first = cloned.next();
                        let second = cloned.next();

                        if first == Some('0') && second != Some('$') {
                            skip_char_and_emit(
                                &mut chars,
                                initial_len,
                                FormatSpecifier::Zero,
                                callback,
                            );
                        }

                        // width
                        match chars.clone().next().unwrap_or_default() {
                            '0'..='9' => {
                                read_integer(&mut chars, initial_len, callback);
                                if chars.clone().next() == Some('$') {
                                    skip_char_and_emit(
                                        &mut chars,
                                        initial_len,
                                        FormatSpecifier::DollarSign,
                                        callback,
                                    );
                                }
                            }
                            'a'..='z' | 'A'..='Z' | '_' => {
                                read_identifier(&mut chars, initial_len, callback);
                                if chars.clone().next() != Some('$') {
                                    continue;
                                }
                                skip_char_and_emit(
                                    &mut chars,
                                    initial_len,
                                    FormatSpecifier::DollarSign,
                                    callback,
                                );
                            }
                            _ => {}
                        }

                        // precision
                        if chars.clone().next() == Some('.') {
                            skip_char_and_emit(
                                &mut chars,
                                initial_len,
                                FormatSpecifier::Dot,
                                callback,
                            );

                            match chars.clone().next().unwrap_or_default() {
                                '*' => {
                                    skip_char_and_emit(
                                        &mut chars,
                                        initial_len,
                                        FormatSpecifier::Asterisk,
                                        callback,
                                    );
                                }
                                '0'..='9' => {
                                    read_integer(&mut chars, initial_len, callback);
                                    if chars.clone().next() == Some('$') {
                                        skip_char_and_emit(
                                            &mut chars,
                                            initial_len,
                                            FormatSpecifier::DollarSign,
                                            callback,
                                        );
                                    }
                                }
                                'a'..='z' | 'A'..='Z' | '_' => {
                                    read_identifier(&mut chars, initial_len, callback);
                                    if chars.clone().next() != Some('$') {
                                        continue;
                                    }
                                    skip_char_and_emit(
                                        &mut chars,
                                        initial_len,
                                        FormatSpecifier::DollarSign,
                                        callback,
                                    );
                                }
                                _ => {
                                    continue;
                                }
                            }
                        }

                        // type
                        match chars.clone().next().unwrap_or_default() {
                            '?' => {
                                skip_char_and_emit(
                                    &mut chars,
                                    initial_len,
                                    FormatSpecifier::QuestionMark,
                                    callback,
                                );
                            }
                            'a'..='z' | 'A'..='Z' | '_' => {
                                read_identifier(&mut chars, initial_len, callback);
                            }
                            _ => {}
                        }
                    }

                    let mut cloned = chars.clone().take(2);
                    let first = cloned.next();
                    let second = cloned.next();
                    if first != Some('}') {
                        continue;
                    }
                    if second == Some('}') {
                        // Escaped format end specifier, `}}`
                        continue;
                    }
                    skip_char_and_emit(&mut chars, initial_len, FormatSpecifier::Close, callback);
                }
                _ => {
                    while let Some(next_char) = chars.clone().next() {
                        match next_char {
                            '{' => break,
                            _ => {}
                        }
                        chars.next();
                    }
                }
            };
        }

        fn skip_char_and_emit<F>(
            chars: &mut std::str::Chars,
            initial_len: usize,
            emit: FormatSpecifier,
            callback: &mut F,
        ) where
            F: FnMut(TextRange, FormatSpecifier),
        {
            let start = initial_len - chars.as_str().len();
            chars.next();
            let end = initial_len - chars.as_str().len();
            callback(
                TextRange::from_to(TextUnit::from_usize(start), TextUnit::from_usize(end)),
                emit,
            );
        }

        fn read_integer<F>(chars: &mut std::str::Chars, initial_len: usize, callback: &mut F)
        where
            F: FnMut(TextRange, FormatSpecifier),
        {
            let start = initial_len - chars.as_str().len();
            chars.next();
            while let Some(next_char) = chars.clone().next() {
                match next_char {
                    '0'..='9' => {
                        chars.next();
                    }
                    _ => {
                        break;
                    }
                }
            }
            let end = initial_len - chars.as_str().len();
            callback(
                TextRange::from_to(TextUnit::from_usize(start), TextUnit::from_usize(end)),
                FormatSpecifier::Integer,
            );
        }
        fn read_identifier<F>(chars: &mut std::str::Chars, initial_len: usize, callback: &mut F)
        where
            F: FnMut(TextRange, FormatSpecifier),
        {
            let start = initial_len - chars.as_str().len();
            chars.next();
            while let Some(next_char) = chars.clone().next() {
                match next_char {
                    'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                        chars.next();
                    }
                    _ => {
                        break;
                    }
                }
            }
            let end = initial_len - chars.as_str().len();
            callback(
                TextRange::from_to(TextUnit::from_usize(start), TextUnit::from_usize(end)),
                FormatSpecifier::Identifier,
            );
        }
    }
}

impl HasFormatSpecifier for String {}
impl HasFormatSpecifier for RawString {}
