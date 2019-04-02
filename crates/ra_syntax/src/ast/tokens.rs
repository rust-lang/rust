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
    pub fn flavor(&self) -> CommentFlavor {
        let text = self.text();
        if text.starts_with("///") {
            CommentFlavor::OuterDoc
        } else if text.starts_with("//!") {
            CommentFlavor::InnerDoc
        } else if text.starts_with("//") {
            CommentFlavor::Line
        } else {
            CommentFlavor::Multiline
        }
    }

    pub fn is_doc_comment(&self) -> bool {
        self.flavor().is_doc_comment()
    }

    pub fn prefix(&self) -> &'static str {
        self.flavor().prefix()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CommentFlavor {
    Line,
    OuterDoc,
    InnerDoc,
    Multiline,
}

impl CommentFlavor {
    pub fn prefix(&self) -> &'static str {
        match *self {
            CommentFlavor::Line => "//",
            CommentFlavor::OuterDoc => "///",
            CommentFlavor::InnerDoc => "//!",
            CommentFlavor::Multiline => "/*",
        }
    }

    pub fn is_doc_comment(&self) -> bool {
        match self {
            CommentFlavor::OuterDoc | CommentFlavor::InnerDoc => true,
            _ => false,
        }
    }
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
