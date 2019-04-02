//! There are many AstNodes, but only a few tokens, so we hand-write them here.

use crate::{
    SyntaxToken,
    SyntaxKind::{COMMENT, WHITESPACE},
    ast::AstToken,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Comment<'a>(SyntaxToken<'a>);

impl<'a> AstToken<'a> for Comment<'a> {
    fn cast(token: SyntaxToken<'a>) -> Option<Self> {
        if token.kind() == COMMENT {
            Some(Comment(token))
        } else {
            None
        }
    }
    fn syntax(&self) -> SyntaxToken<'a> {
        self.0
    }
}

impl<'a> Comment<'a> {
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
    use {CommentShape::*, CommentPlacement::*};
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

pub struct Whitespace<'a>(SyntaxToken<'a>);

impl<'a> AstToken<'a> for Whitespace<'a> {
    fn cast(token: SyntaxToken<'a>) -> Option<Self> {
        if token.kind() == WHITESPACE {
            Some(Whitespace(token))
        } else {
            None
        }
    }
    fn syntax(&self) -> SyntaxToken<'a> {
        self.0
    }
}

impl<'a> Whitespace<'a> {
    pub fn spans_multiple_lines(&self) -> bool {
        let text = self.text();
        text.find('\n').map_or(false, |idx| text[idx + 1..].contains('\n'))
    }
}
