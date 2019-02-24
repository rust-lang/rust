//! This module defines Concrete Syntax Tree (CST), used by rust-analyzer.
//!
//! The CST includes comments and whitespace, provides a single node type,
//! `SyntaxNode`, and a basic traversal API (parent, children, siblings).
//!
//! The *real* implementation is in the (language-agnostic) `rowan` crate, this
//! modules just wraps its API.

use std::{
    fmt::{self, Write},
    borrow::Borrow,
};

use ra_parser::ParseError;
use rowan::{Types, TransparentNewType, GreenNodeBuilder};

use crate::{
    SmolStr, SyntaxKind, TextUnit, TextRange, SyntaxText, SourceFile, AstNode,
    syntax_error::{SyntaxError, SyntaxErrorKind},
};

pub use rowan::WalkEvent;

#[derive(Debug, Clone, Copy)]
pub enum RaTypes {}
impl Types for RaTypes {
    type Kind = SyntaxKind;
    type RootData = Vec<SyntaxError>;
}

pub(crate) type GreenNode = rowan::GreenNode<RaTypes>;

/// Marker trait for CST and AST nodes
pub trait SyntaxNodeWrapper: TransparentNewType<Repr = rowan::SyntaxNode<RaTypes>> {}
impl<T: TransparentNewType<Repr = rowan::SyntaxNode<RaTypes>>> SyntaxNodeWrapper for T {}

/// An owning smart pointer for CST or AST node.
#[derive(PartialEq, Eq, Hash)]
pub struct TreeArc<T: SyntaxNodeWrapper>(pub(crate) rowan::TreeArc<RaTypes, T>);

impl<T: SyntaxNodeWrapper> Borrow<T> for TreeArc<T> {
    fn borrow(&self) -> &T {
        &*self
    }
}

impl<T> TreeArc<T>
where
    T: SyntaxNodeWrapper,
{
    pub(crate) fn cast<U>(this: TreeArc<T>) -> TreeArc<U>
    where
        U: SyntaxNodeWrapper,
    {
        TreeArc(rowan::TreeArc::cast(this.0))
    }
}

impl<T> std::ops::Deref for TreeArc<T>
where
    T: SyntaxNodeWrapper,
{
    type Target = T;
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<T> PartialEq<T> for TreeArc<T>
where
    T: SyntaxNodeWrapper,
    T: PartialEq<T>,
{
    fn eq(&self, other: &T) -> bool {
        let t: &T = self;
        t == other
    }
}

impl<T> Clone for TreeArc<T>
where
    T: SyntaxNodeWrapper,
{
    fn clone(&self) -> TreeArc<T> {
        TreeArc(self.0.clone())
    }
}

impl<T> fmt::Debug for TreeArc<T>
where
    T: SyntaxNodeWrapper,
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, fmt)
    }
}

#[derive(PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SyntaxNode(pub(crate) rowan::SyntaxNode<RaTypes>);
unsafe impl TransparentNewType for SyntaxNode {
    type Repr = rowan::SyntaxNode<RaTypes>;
}

impl ToOwned for SyntaxNode {
    type Owned = TreeArc<SyntaxNode>;
    fn to_owned(&self) -> TreeArc<SyntaxNode> {
        let ptr = TreeArc(self.0.to_owned());
        TreeArc::cast(ptr)
    }
}

impl fmt::Debug for SyntaxNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}@{:?}", self.kind(), self.range())?;
        if has_short_text(self.kind()) {
            write!(fmt, " \"{}\"", self.text())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Next,
    Prev,
}

impl SyntaxNode {
    pub(crate) fn new(green: GreenNode, errors: Vec<SyntaxError>) -> TreeArc<SyntaxNode> {
        let ptr = TreeArc(rowan::SyntaxNode::new(green, errors));
        TreeArc::cast(ptr)
    }

    pub fn kind(&self) -> SyntaxKind {
        self.0.kind()
    }

    pub fn range(&self) -> TextRange {
        self.0.range()
    }

    pub fn text(&self) -> SyntaxText {
        SyntaxText::new(self)
    }

    pub fn is_leaf(&self) -> bool {
        self.0.is_leaf()
    }

    pub fn leaf_text(&self) -> Option<&SmolStr> {
        self.0.leaf_text()
    }

    pub fn parent(&self) -> Option<&SyntaxNode> {
        self.0.parent().map(SyntaxNode::from_repr)
    }

    pub fn first_child(&self) -> Option<&SyntaxNode> {
        self.0.first_child().map(SyntaxNode::from_repr)
    }

    pub fn last_child(&self) -> Option<&SyntaxNode> {
        self.0.last_child().map(SyntaxNode::from_repr)
    }

    pub fn next_sibling(&self) -> Option<&SyntaxNode> {
        self.0.next_sibling().map(SyntaxNode::from_repr)
    }

    pub fn prev_sibling(&self) -> Option<&SyntaxNode> {
        self.0.prev_sibling().map(SyntaxNode::from_repr)
    }

    pub fn children(&self) -> SyntaxNodeChildren {
        SyntaxNodeChildren(self.0.children())
    }

    pub fn ancestors(&self) -> impl Iterator<Item = &SyntaxNode> {
        crate::algo::generate(Some(self), |&node| node.parent())
    }

    pub fn descendants(&self) -> impl Iterator<Item = &SyntaxNode> {
        self.preorder().filter_map(|event| match event {
            WalkEvent::Enter(node) => Some(node),
            WalkEvent::Leave(_) => None,
        })
    }

    pub fn siblings(&self, direction: Direction) -> impl Iterator<Item = &SyntaxNode> {
        crate::algo::generate(Some(self), move |&node| match direction {
            Direction::Next => node.next_sibling(),
            Direction::Prev => node.prev_sibling(),
        })
    }

    pub fn preorder(&self) -> impl Iterator<Item = WalkEvent<&SyntaxNode>> {
        self.0.preorder().map(|event| match event {
            WalkEvent::Enter(n) => WalkEvent::Enter(SyntaxNode::from_repr(n)),
            WalkEvent::Leave(n) => WalkEvent::Leave(SyntaxNode::from_repr(n)),
        })
    }

    pub fn memory_size_of_subtree(&self) -> usize {
        self.0.memory_size_of_subtree()
    }

    pub fn debug_dump(&self) -> String {
        let mut errors: Vec<_> = match self.ancestors().find_map(SourceFile::cast) {
            Some(file) => file.errors(),
            None => self.root_data().to_vec(),
        };
        errors.sort_by_key(|e| e.offset());
        let mut err_pos = 0;
        let mut level = 0;
        let mut buf = String::new();
        macro_rules! indent {
            () => {
                for _ in 0..level {
                    buf.push_str("  ");
                }
            };
        }

        for event in self.preorder() {
            match event {
                WalkEvent::Enter(node) => {
                    indent!();
                    writeln!(buf, "{:?}", node).unwrap();
                    if node.first_child().is_none() {
                        let off = node.range().end();
                        while err_pos < errors.len() && errors[err_pos].offset() <= off {
                            indent!();
                            writeln!(buf, "err: `{}`", errors[err_pos]).unwrap();
                            err_pos += 1;
                        }
                    }
                    level += 1;
                }
                WalkEvent::Leave(_) => level -= 1,
            }
        }

        assert_eq!(level, 0);
        for err in errors[err_pos..].iter() {
            writeln!(buf, "err: `{}`", err).unwrap();
        }

        buf
    }

    pub(crate) fn root_data(&self) -> &Vec<SyntaxError> {
        self.0.root_data()
    }

    pub(crate) fn replace_with(&self, replacement: GreenNode) -> GreenNode {
        self.0.replace_self(replacement)
    }
}

#[derive(Debug)]
pub struct SyntaxNodeChildren<'a>(rowan::SyntaxNodeChildren<'a, RaTypes>);

impl<'a> Iterator for SyntaxNodeChildren<'a> {
    type Item = &'a SyntaxNode;

    fn next(&mut self) -> Option<&'a SyntaxNode> {
        self.0.next().map(SyntaxNode::from_repr)
    }
}

fn has_short_text(kind: SyntaxKind) -> bool {
    use crate::SyntaxKind::*;
    match kind {
        IDENT | LIFETIME | INT_NUMBER | FLOAT_NUMBER => true,
        _ => false,
    }
}

pub struct SyntaxTreeBuilder {
    errors: Vec<SyntaxError>,
    inner: GreenNodeBuilder<RaTypes>,
}

impl Default for SyntaxTreeBuilder {
    fn default() -> SyntaxTreeBuilder {
        SyntaxTreeBuilder { errors: Vec::new(), inner: GreenNodeBuilder::new() }
    }
}

impl SyntaxTreeBuilder {
    pub(crate) fn finish_raw(self) -> (GreenNode, Vec<SyntaxError>) {
        let green = self.inner.finish();
        (green, self.errors)
    }

    pub fn finish(self) -> TreeArc<SyntaxNode> {
        let (green, errors) = self.finish_raw();
        let node = SyntaxNode::new(green, errors);
        if cfg!(debug_assertions) {
            crate::validation::validate_block_structure(&node);
        }
        node
    }

    pub fn leaf(&mut self, kind: SyntaxKind, text: SmolStr) {
        self.inner.leaf(kind, text)
    }

    pub fn start_branch(&mut self, kind: SyntaxKind) {
        self.inner.start_internal(kind)
    }

    pub fn finish_branch(&mut self) {
        self.inner.finish_internal()
    }

    pub fn error(&mut self, error: ParseError, text_pos: TextUnit) {
        let error = SyntaxError::new(SyntaxErrorKind::ParseError(error), text_pos);
        self.errors.push(error)
    }
}
