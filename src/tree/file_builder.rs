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
use super::{File, NodeData, NodeIdx, SyntaxErrorData};
use SError;

pub(crate) trait Sink {
    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit);
    fn start_internal(&mut self, kind: SyntaxKind);
    fn finish_internal(&mut self);
    fn error(&mut self, err: ErrorMsg);
}

#[derive(Debug)]
pub(crate) struct FileBuilder {
    text: String,
    nodes: Vec<NodeData>,
    errors: Vec<SyntaxErrorData>,
    in_progress: Vec<(NodeIdx, Option<NodeIdx>)>,
    // (parent, last_child)
    pos: TextUnit,
}

impl Sink for FileBuilder {
    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit) {
        let leaf = NodeData {
            kind,
            range: TextRange::offset_len(self.pos, len),
            parent: None,
            first_child: None,
            next_sibling: None,
        };
        self.pos += len;
        let id = self.push_child(leaf);
        self.add_len(id);
    }

    fn start_internal(&mut self, kind: SyntaxKind) {
        let node = NodeData {
            kind,
            range: TextRange::offset_len(self.pos, 0.into()),
            parent: None,
            first_child: None,
            next_sibling: None,
        };
        let id = if self.in_progress.is_empty() {
            self.new_node(node)
        } else {
            self.push_child(node)
        };
        self.in_progress.push((id, None))
    }

    fn finish_internal(&mut self) {
        let (id, _) = self.in_progress
            .pop()
            .expect("trying to complete a node, but there are no in-progress nodes");
        if !self.in_progress.is_empty() {
            self.add_len(id);
        }
    }

    fn error(&mut self, err: ErrorMsg) {
        let &(node, after_child) = self.in_progress.last().unwrap();
        self.errors.push(SyntaxErrorData {
            node,
            message: err.msg,
            after_child,
        })
    }
}

impl FileBuilder {
    pub fn new(text: String) -> FileBuilder {
        FileBuilder {
            text,
            nodes: Vec::new(),
            errors: Vec::new(),
            in_progress: Vec::new(),
            pos: 0.into(),
        }
    }

    pub fn finish(self) -> File {
        assert!(
            self.in_progress.is_empty(),
            "some nodes in FileBuilder are unfinished: {:?}",
            self.in_progress
                .iter()
                .map(|&(idx, _)| self.nodes[idx].kind)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            self.pos,
            (self.text.len() as u32).into(),
            "nodes in FileBuilder do not cover the whole file"
        );
        File {
            text: self.text,
            nodes: self.nodes,
            errors: self.errors,
        }
    }

    fn new_node(&mut self, data: NodeData) -> NodeIdx {
        let id = NodeIdx(self.nodes.len() as u32);
        self.nodes.push(data);
        id
    }

    fn push_child(&mut self, mut child: NodeData) -> NodeIdx {
        child.parent = Some(self.current_id());
        let id = self.new_node(child);
        {
            let (parent, sibling) = *self.in_progress.last().unwrap();
            let slot = if let Some(idx) = sibling {
                &mut self.nodes[idx].next_sibling
            } else {
                &mut self.nodes[parent].first_child
            };
            fill(slot, id);
        }
        self.in_progress.last_mut().unwrap().1 = Some(id);
        id
    }

    fn add_len(&mut self, child: NodeIdx) {
        let range = self.nodes[child].range;
        grow(&mut self.current_parent().range, range);
    }

    fn current_id(&self) -> NodeIdx {
        self.in_progress.last().unwrap().0
    }

    fn current_parent(&mut self) -> &mut NodeData {
        let idx = self.current_id();
        &mut self.nodes[idx]
    }
}

fn fill<T>(slot: &mut Option<T>, value: T) {
    assert!(slot.is_none());
    *slot = Some(value);
}

fn grow(left: &mut TextRange, right: TextRange) {
    assert_eq!(left.end(), right.start());
    *left = TextRange::from_to(left.start(), right.end())
}

#[derive(Default)]
pub(crate) struct ErrorMsg {
    pub(crate) msg: String,
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

    fn error(&mut self, err: ErrorMsg) {
        self.errors.push(SError { message: err.msg, offset: self.pos })
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

