//! There are many AstNodes, but only a few tokens, so we hand-write them here.

use crate::{
    ast::AstToken,
    SyntaxKind::{COMMENT, RAW_STRING, STRING, WHITESPACE},
    SyntaxToken, TextRange, TextUnit,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Comment(SyntaxToken);

impl AstToken for Comment {
    fn cast(token: SyntaxToken) -> Option<Self> {
        match token.kind() {
            COMMENT => Some(Comment(token)),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxToken {
        &self.0
    }
}

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

pub struct Whitespace(SyntaxToken);

impl AstToken for Whitespace {
    fn cast(token: SyntaxToken) -> Option<Self> {
        match token.kind() {
            WHITESPACE => Some(Whitespace(token)),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxToken {
        &self.0
    }
}

impl Whitespace {
    pub fn spans_multiple_lines(&self) -> bool {
        let text = self.text();
        text.find('\n').map_or(false, |idx| text[idx + 1..].contains('\n'))
    }
}

pub struct String(SyntaxToken);

impl AstToken for String {
    fn cast(token: SyntaxToken) -> Option<Self> {
        match token.kind() {
            STRING => Some(String(token)),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxToken {
        &self.0
    }
}

impl String {
    pub fn value(&self) -> Option<std::string::String> {
        let text = self.text().as_str();
        let usual_string_range = find_usual_string_range(text)?;
        let start_of_inside = usual_string_range.start().to_usize() + 1;
        let end_of_inside = usual_string_range.end().to_usize();
        let inside_str = &text[start_of_inside..end_of_inside];

        let mut buf = std::string::String::with_capacity(inside_str.len());
        let mut has_error = false;
        rustc_lexer::unescape::unescape_str(inside_str, &mut |_, unescaped_char| {
            match unescaped_char {
                Ok(c) => buf.push(c),
                Err(_) => has_error = true,
            }
        });

        if has_error {
            return None;
        }
        Some(buf)
    }
}

pub struct RawString(SyntaxToken);

impl AstToken for RawString {
    fn cast(token: SyntaxToken) -> Option<Self> {
        match token.kind() {
            RAW_STRING => Some(RawString(token)),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxToken {
        &self.0
    }
}

impl RawString {
    pub fn value(&self) -> Option<std::string::String> {
        let text = self.text().as_str();
        let usual_string_range = find_usual_string_range(text)?;
        let start_of_inside = usual_string_range.start().to_usize() + 1;
        let end_of_inside = usual_string_range.end().to_usize();
        let inside_str = &text[start_of_inside..end_of_inside];
        Some(inside_str.to_string())
    }
}

fn find_usual_string_range(s: &str) -> Option<TextRange> {
    let left_quote = s.find('"')?;
    let right_quote = s.rfind('"')?;
    if left_quote == right_quote {
        // `s` only contains one quote
        None
    } else {
        Some(TextRange::from_to(
            TextUnit::from(left_quote as u32),
            TextUnit::from(right_quote as u32),
        ))
    }
}
