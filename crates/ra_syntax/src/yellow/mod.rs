mod builder;
mod syntax_text;

use std::{
    fmt,
    hash::{Hash, Hasher},
};
use rowan::Types;
use {SyntaxKind, TextUnit, TextRange, SmolStr};
use self::syntax_text::SyntaxText;

pub use rowan::{TreeRoot};
pub(crate) use self::builder::GreenBuilder;

#[derive(Debug, Clone, Copy)]
pub enum RaTypes {}
impl Types for RaTypes {
    type Kind = SyntaxKind;
    type RootData = Vec<SyntaxError>;
}

pub type OwnedRoot = ::rowan::OwnedRoot<RaTypes>;
pub type RefRoot<'a> = ::rowan::RefRoot<'a, RaTypes>;

pub type GreenNode = ::rowan::GreenNode<RaTypes>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SyntaxError {
    pub msg: String,
    pub offset: TextUnit,
}

#[derive(Clone, Copy)]
pub struct SyntaxNode<R: TreeRoot<RaTypes> = OwnedRoot>(
    ::rowan::SyntaxNode<RaTypes, R>,
);
pub type SyntaxNodeRef<'a> = SyntaxNode<RefRoot<'a>>;

impl<R1, R2> PartialEq<SyntaxNode<R1>> for SyntaxNode<R2>
where
    R1: TreeRoot<RaTypes>,
    R2: TreeRoot<RaTypes>,
{
    fn eq(&self, other: &SyntaxNode<R1>) -> bool {
        self.0 == other.0
    }
}

impl<R: TreeRoot<RaTypes>> Eq for SyntaxNode<R> {}
impl<R: TreeRoot<RaTypes>> Hash for SyntaxNode<R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl SyntaxNode {
    pub(crate) fn new(green: GreenNode, errors: Vec<SyntaxError>) -> SyntaxNode {
        SyntaxNode(::rowan::SyntaxNode::new(green, errors))
    }
}
impl<'a> SyntaxNodeRef<'a> {
    pub fn leaf_text(self) -> Option<&'a SmolStr> {
        self.0.leaf_text()
    }
}

impl<R: TreeRoot<RaTypes>> SyntaxNode<R> {
    pub(crate) fn root_data(&self) -> &Vec<SyntaxError> {
        self.0.root_data()
    }
    pub(crate) fn replace_with(&self, replacement: GreenNode) -> GreenNode {
        self.0.replace_with(replacement)
    }
    pub fn borrowed<'a>(&'a self) -> SyntaxNode<RefRoot<'a>> {
        SyntaxNode(self.0.borrowed())
    }
    pub fn owned(&self) -> SyntaxNode<OwnedRoot> {
        SyntaxNode(self.0.owned())
    }
    pub fn kind(&self) -> SyntaxKind {
        self.0.kind()
    }
    pub fn range(&self) -> TextRange {
        self.0.range()
    }
    pub fn text(&self) -> SyntaxText {
        SyntaxText::new(self.borrowed())
    }
    pub fn is_leaf(&self) -> bool {
        self.0.is_leaf()
    }
    pub fn parent(&self) -> Option<SyntaxNode<R>> {
        self.0.parent().map(SyntaxNode)
    }
    pub fn first_child(&self) -> Option<SyntaxNode<R>> {
        self.0.first_child().map(SyntaxNode)
    }
    pub fn last_child(&self) -> Option<SyntaxNode<R>> {
        self.0.last_child().map(SyntaxNode)
    }
    pub fn next_sibling(&self) -> Option<SyntaxNode<R>> {
        self.0.next_sibling().map(SyntaxNode)
    }
    pub fn prev_sibling(&self) -> Option<SyntaxNode<R>> {
        self.0.prev_sibling().map(SyntaxNode)
    }
    pub fn children(&self) -> SyntaxNodeChildren<R> {
        SyntaxNodeChildren(self.0.children())
    }
}

impl<R: TreeRoot<RaTypes>> fmt::Debug for SyntaxNode<R> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}@{:?}", self.kind(), self.range())?;
        if has_short_text(self.kind()) {
            write!(fmt, " \"{}\"", self.text())?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SyntaxNodeChildren<R: TreeRoot<RaTypes>>(
    ::rowan::SyntaxNodeChildren<RaTypes, R>
);

impl<R: TreeRoot<RaTypes>> Iterator for SyntaxNodeChildren<R> {
    type Item = SyntaxNode<R>;

    fn next(&mut self) -> Option<SyntaxNode<R>> {
        self.0.next().map(SyntaxNode)
    }
}


fn has_short_text(kind: SyntaxKind) -> bool {
    use SyntaxKind::*;
    match kind {
        IDENT | LIFETIME | INT_NUMBER | FLOAT_NUMBER => true,
        _ => false,
    }
}
