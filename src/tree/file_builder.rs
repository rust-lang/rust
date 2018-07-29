//! This module provides a way to construct a `File`.
//! It is intended to be completely decoupled from the
//! parser, so as to allow to evolve the tree representation
//! and the parser algorithm independently.
//!
//! The `Sink` trait is the bridge between the parser and the
//! tree builder: the parser produces a stream of events like
//! `start node`, `finish node`, and `FileBuilder` converts
//! this stream to a real tree.
use std::sync::Arc;
use {
    SyntaxKind, TextRange, TextUnit,
    yellow::GreenNode
};
use SError;

pub(crate) trait Sink {
    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit);
    fn start_internal(&mut self, kind: SyntaxKind);
    fn finish_internal(&mut self);
    fn error(&mut self, err: String);
}

pub(crate) struct GreenBuilder {
    text: String,
    stack: Vec<GreenNode>,
    pos: TextUnit,
    root: Option<GreenNode>,
    errors: Vec<SError>,
}

impl GreenBuilder {
    pub(crate) fn new(text: String) -> GreenBuilder {
        GreenBuilder {
            text,
            stack: Vec::new(),
            pos: 0.into(),
            root: None,
            errors: Vec::new(),
        }
    }

    pub(crate) fn finish(self) -> (GreenNode, Vec<SError>) {
        (self.root.unwrap(), self.errors)
    }
}

impl Sink for GreenBuilder {
    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit) {
        let range = TextRange::offset_len(self.pos, len);
        self.pos += len;
        let text = self.text[range].to_owned();
        let parent = self.stack.last_mut().unwrap();
        if kind.is_trivia() {
            parent.push_trivia(kind, text);
        } else {
            let node = GreenNode::new_leaf(kind, text);
            parent.push_child(Arc::new(node));
        }
    }

    fn start_internal(&mut self, kind: SyntaxKind) {
        self.stack.push(GreenNode::new_branch(kind))
    }

    fn finish_internal(&mut self) {
        let node = self.stack.pop().unwrap();
        if let Some(parent) = self.stack.last_mut() {
            parent.push_child(Arc::new(node))
        } else {
            self.root = Some(node);
        }
    }

    fn error(&mut self, message: String) {
        self.errors.push(SError { message, offset: self.pos })
    }
}
impl SyntaxKind {
    fn is_trivia(self) -> bool {
        match self {
            SyntaxKind::WHITESPACE | SyntaxKind::DOC_COMMENT | SyntaxKind::COMMENT => true,
            _ => false
        }
    }
}

