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
