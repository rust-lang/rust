//! This module defines Concrete Syntax Tree (CST), used by rust-analyzer.
//!
//! The CST includes comments and whitespace, provides a single node type,
//! `SyntaxNode`, and a basic traversal API (parent, children, siblings).
//!
//! The *real* implementation is in the (language-agnostic) `rowan` crate, this
//! modules just wraps its API.

use std::{
    fmt::{self, Write},
    iter::successors,
    ops::RangeInclusive,
};

use ra_parser::ParseError;
use rowan::GreenNodeBuilder;

use crate::{
    syntax_error::{SyntaxError, SyntaxErrorKind},
    AstNode, Parse, SmolStr, SourceFile, SyntaxKind, SyntaxNodePtr, SyntaxText, TextRange,
    TextUnit,
};

pub use rowan::WalkEvent;
pub(crate) use rowan::{GreenNode, GreenToken};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum InsertPosition<T> {
    First,
    Last,
    Before(T),
    After(T),
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct SyntaxNode(pub(crate) rowan::cursor::SyntaxNode);

impl fmt::Debug for SyntaxNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}@{:?}", self.kind(), self.range())
    }
}

impl fmt::Display for SyntaxNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.text(), fmt)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Next,
    Prev,
}

impl SyntaxNode {
    pub(crate) fn new(green: GreenNode) -> SyntaxNode {
        let inner = rowan::cursor::SyntaxNode::new_root(green);
        SyntaxNode(inner)
    }

    pub fn kind(&self) -> SyntaxKind {
        self.0.kind().0.into()
    }

    pub fn range(&self) -> TextRange {
        self.0.text_range()
    }

    pub fn text(&self) -> SyntaxText {
        SyntaxText::new(self)
    }

    pub fn parent(&self) -> Option<SyntaxNode> {
        self.0.parent().map(SyntaxNode)
    }

    pub fn first_child(&self) -> Option<SyntaxNode> {
        self.0.first_child().map(SyntaxNode)
    }

    pub fn first_child_or_token(&self) -> Option<SyntaxElement> {
        self.0.first_child_or_token().map(SyntaxElement::new)
    }

    pub fn last_child(&self) -> Option<SyntaxNode> {
        self.0.last_child().map(SyntaxNode)
    }

    pub fn last_child_or_token(&self) -> Option<SyntaxElement> {
        self.0.last_child_or_token().map(SyntaxElement::new)
    }

    pub fn next_sibling(&self) -> Option<SyntaxNode> {
        self.0.next_sibling().map(SyntaxNode)
    }

    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.0.next_sibling_or_token().map(SyntaxElement::new)
    }

    pub fn prev_sibling(&self) -> Option<SyntaxNode> {
        self.0.prev_sibling().map(SyntaxNode)
    }

    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.0.prev_sibling_or_token().map(SyntaxElement::new)
    }

    pub fn children(&self) -> SyntaxNodeChildren {
        SyntaxNodeChildren(self.0.children())
    }

    pub fn children_with_tokens(&self) -> SyntaxElementChildren {
        SyntaxElementChildren(self.0.children_with_tokens())
    }

    pub fn first_token(&self) -> Option<SyntaxToken> {
        self.0.first_token().map(SyntaxToken)
    }

    pub fn last_token(&self) -> Option<SyntaxToken> {
        self.0.last_token().map(SyntaxToken)
    }

    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        successors(Some(self.clone()), |node| node.parent())
    }

    pub fn descendants(&self) -> impl Iterator<Item = SyntaxNode> {
        self.preorder().filter_map(|event| match event {
            WalkEvent::Enter(node) => Some(node),
            WalkEvent::Leave(_) => None,
        })
    }

    pub fn descendants_with_tokens(&self) -> impl Iterator<Item = SyntaxElement> {
        self.preorder_with_tokens().filter_map(|event| match event {
            WalkEvent::Enter(it) => Some(it),
            WalkEvent::Leave(_) => None,
        })
    }

    pub fn siblings(&self, direction: Direction) -> impl Iterator<Item = SyntaxNode> {
        successors(Some(self.clone()), move |node| match direction {
            Direction::Next => node.next_sibling(),
            Direction::Prev => node.prev_sibling(),
        })
    }

    pub fn siblings_with_tokens(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = SyntaxElement> {
        let me: SyntaxElement = self.clone().into();
        successors(Some(me), move |el| match direction {
            Direction::Next => el.next_sibling_or_token(),
            Direction::Prev => el.prev_sibling_or_token(),
        })
    }

    pub fn preorder(&self) -> impl Iterator<Item = WalkEvent<SyntaxNode>> {
        self.0.preorder().map(|event| match event {
            WalkEvent::Enter(n) => WalkEvent::Enter(SyntaxNode(n)),
            WalkEvent::Leave(n) => WalkEvent::Leave(SyntaxNode(n)),
        })
    }

    pub fn preorder_with_tokens(&self) -> impl Iterator<Item = WalkEvent<SyntaxElement>> {
        self.0.preorder_with_tokens().map(|event| match event {
            WalkEvent::Enter(n) => WalkEvent::Enter(SyntaxElement::new(n)),
            WalkEvent::Leave(n) => WalkEvent::Leave(SyntaxElement::new(n)),
        })
    }

    pub fn memory_size_of_subtree(&self) -> usize {
        0
    }

    pub fn debug_dump(&self) -> String {
        let mut level = 0;
        let mut buf = String::new();

        for event in self.preorder_with_tokens() {
            match event {
                WalkEvent::Enter(element) => {
                    for _ in 0..level {
                        buf.push_str("  ");
                    }
                    match element {
                        SyntaxElement::Node(node) => writeln!(buf, "{:?}", node).unwrap(),
                        SyntaxElement::Token(token) => writeln!(buf, "{:?}", token).unwrap(),
                    }
                    level += 1;
                }
                WalkEvent::Leave(_) => level -= 1,
            }
        }

        assert_eq!(level, 0);

        buf
    }

    pub(crate) fn replace_with(&self, replacement: GreenNode) -> GreenNode {
        self.0.replace_with(replacement)
    }

    /// Adds specified children (tokens or nodes) to the current node at the
    /// specific position.
    ///
    /// This is a type-unsafe low-level editing API, if you need to use it,
    /// prefer to create a type-safe abstraction on top of it instead.
    pub fn insert_children(
        &self,
        position: InsertPosition<SyntaxElement>,
        to_insert: impl Iterator<Item = SyntaxElement>,
    ) -> SyntaxNode {
        let mut delta = TextUnit::default();
        let to_insert = to_insert.map(|element| {
            delta += element.text_len();
            to_green_element(element)
        });

        let old_children = self.0.green().children();

        let new_children = match &position {
            InsertPosition::First => {
                to_insert.chain(old_children.iter().cloned()).collect::<Box<[_]>>()
            }
            InsertPosition::Last => {
                old_children.iter().cloned().chain(to_insert).collect::<Box<[_]>>()
            }
            InsertPosition::Before(anchor) | InsertPosition::After(anchor) => {
                let take_anchor = if let InsertPosition::After(_) = position { 1 } else { 0 };
                let split_at = self.position_of_child(anchor.clone()) + take_anchor;
                let (before, after) = old_children.split_at(split_at);
                before
                    .iter()
                    .cloned()
                    .chain(to_insert)
                    .chain(after.iter().cloned())
                    .collect::<Box<[_]>>()
            }
        };

        self.with_children(new_children)
    }

    /// Replaces all nodes in `to_delete` with nodes from `to_insert`
    ///
    /// This is a type-unsafe low-level editing API, if you need to use it,
    /// prefer to create a type-safe abstraction on top of it instead.
    pub fn replace_children(
        &self,
        to_delete: RangeInclusive<SyntaxElement>,
        to_insert: impl Iterator<Item = SyntaxElement>,
    ) -> SyntaxNode {
        let start = self.position_of_child(to_delete.start().clone());
        let end = self.position_of_child(to_delete.end().clone());
        let old_children = self.0.green().children();

        let new_children = old_children[..start]
            .iter()
            .cloned()
            .chain(to_insert.map(to_green_element))
            .chain(old_children[end + 1..].iter().cloned())
            .collect::<Box<[_]>>();
        self.with_children(new_children)
    }

    fn with_children(&self, new_children: Box<[rowan::GreenElement]>) -> SyntaxNode {
        let len = new_children.iter().map(|it| it.text_len()).sum::<TextUnit>();
        let new_node = GreenNode::new(rowan::SyntaxKind(self.kind() as u16), new_children);
        let new_file_node = self.replace_with(new_node);
        let file = SourceFile::new(new_file_node);

        // FIXME: use a more elegant way to re-fetch the node (#1185), make
        // `range` private afterwards
        let mut ptr = SyntaxNodePtr::new(self);
        ptr.range = TextRange::offset_len(ptr.range().start(), len);
        ptr.to_node(file.syntax()).to_owned()
    }

    fn position_of_child(&self, child: SyntaxElement) -> usize {
        self.children_with_tokens()
            .position(|it| it == child)
            .expect("element is not a child of current element")
    }
}

fn to_green_element(element: SyntaxElement) -> rowan::GreenElement {
    match element {
        SyntaxElement::Node(node) => node.0.green().clone().into(),
        SyntaxElement::Token(tok) => {
            GreenToken::new(rowan::SyntaxKind(tok.kind() as u16), tok.text().clone()).into()
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SyntaxToken(pub(crate) rowan::cursor::SyntaxToken);

//FIXME: always output text
impl fmt::Debug for SyntaxToken {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}@{:?}", self.kind(), self.range())?;
        if self.text().len() < 25 {
            return write!(fmt, " {:?}", self.text());
        }
        let text = self.text().as_str();
        for idx in 21..25 {
            if text.is_char_boundary(idx) {
                let text = format!("{} ...", &text[..idx]);
                return write!(fmt, " {:?}", text);
            }
        }
        unreachable!()
    }
}

impl fmt::Display for SyntaxToken {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.text(), fmt)
    }
}

impl SyntaxToken {
    pub fn kind(&self) -> SyntaxKind {
        self.0.kind().0.into()
    }

    pub fn text(&self) -> &SmolStr {
        self.0.text()
    }

    pub fn range(&self) -> TextRange {
        self.0.text_range()
    }

    pub fn parent(&self) -> SyntaxNode {
        SyntaxNode(self.0.parent())
    }

    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.0.next_sibling_or_token().map(SyntaxElement::new)
    }

    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.0.prev_sibling_or_token().map(SyntaxElement::new)
    }

    pub fn siblings_with_tokens(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = SyntaxElement> {
        let me: SyntaxElement = self.clone().into();
        successors(Some(me), move |el| match direction {
            Direction::Next => el.next_sibling_or_token(),
            Direction::Prev => el.prev_sibling_or_token(),
        })
    }

    pub fn next_token(&self) -> Option<SyntaxToken> {
        self.0.next_token().map(SyntaxToken)
    }

    pub fn prev_token(&self) -> Option<SyntaxToken> {
        self.0.prev_token().map(SyntaxToken)
    }

    pub(crate) fn replace_with(&self, new_token: GreenToken) -> GreenNode {
        self.0.replace_with(new_token)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum SyntaxElement {
    Node(SyntaxNode),
    Token(SyntaxToken),
}

impl From<SyntaxNode> for SyntaxElement {
    fn from(node: SyntaxNode) -> Self {
        SyntaxElement::Node(node)
    }
}

impl From<SyntaxToken> for SyntaxElement {
    fn from(token: SyntaxToken) -> Self {
        SyntaxElement::Token(token)
    }
}

impl fmt::Display for SyntaxElement {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SyntaxElement::Node(it) => fmt::Display::fmt(it, fmt),
            SyntaxElement::Token(it) => fmt::Display::fmt(it, fmt),
        }
    }
}

impl SyntaxElement {
    pub(crate) fn new(el: rowan::cursor::SyntaxElement) -> Self {
        match el {
            rowan::cursor::SyntaxElement::Node(it) => SyntaxElement::Node(SyntaxNode(it)),
            rowan::cursor::SyntaxElement::Token(it) => SyntaxElement::Token(SyntaxToken(it)),
        }
    }

    pub fn kind(&self) -> SyntaxKind {
        match self {
            SyntaxElement::Node(it) => it.kind(),
            SyntaxElement::Token(it) => it.kind(),
        }
    }

    pub fn as_node(&self) -> Option<&SyntaxNode> {
        match self {
            SyntaxElement::Node(node) => Some(node),
            SyntaxElement::Token(_) => None,
        }
    }

    pub fn as_token(&self) -> Option<&SyntaxToken> {
        match self {
            SyntaxElement::Node(_) => None,
            SyntaxElement::Token(token) => Some(token),
        }
    }

    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        match self {
            SyntaxElement::Node(it) => it.next_sibling_or_token(),
            SyntaxElement::Token(it) => it.next_sibling_or_token(),
        }
    }

    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        match self {
            SyntaxElement::Node(it) => it.prev_sibling_or_token(),
            SyntaxElement::Token(it) => it.prev_sibling_or_token(),
        }
    }

    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        match self {
            SyntaxElement::Node(it) => it.clone(),
            SyntaxElement::Token(it) => it.parent(),
        }
        .ancestors()
    }

    pub fn range(&self) -> TextRange {
        match self {
            SyntaxElement::Node(it) => it.range(),
            SyntaxElement::Token(it) => it.range(),
        }
    }

    fn text_len(&self) -> TextUnit {
        match self {
            SyntaxElement::Node(node) => node.0.green().text_len(),
            SyntaxElement::Token(token) => TextUnit::of_str(token.0.text()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxNodeChildren(rowan::cursor::SyntaxNodeChildren);

impl Iterator for SyntaxNodeChildren {
    type Item = SyntaxNode;
    fn next(&mut self) -> Option<SyntaxNode> {
        self.0.next().map(SyntaxNode)
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxElementChildren(rowan::cursor::SyntaxElementChildren);

impl Iterator for SyntaxElementChildren {
    type Item = SyntaxElement;
    fn next(&mut self) -> Option<SyntaxElement> {
        self.0.next().map(SyntaxElement::new)
    }
}

pub struct SyntaxTreeBuilder {
    errors: Vec<SyntaxError>,
    inner: GreenNodeBuilder,
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

    pub fn finish(self) -> Parse<SyntaxNode> {
        let (green, errors) = self.finish_raw();
        let node = SyntaxNode::new(green);
        if cfg!(debug_assertions) {
            crate::validation::validate_block_structure(&node);
        }
        Parse::new(node.0.green().clone(), errors)
    }

    pub fn token(&mut self, kind: SyntaxKind, text: SmolStr) {
        self.inner.token(rowan::SyntaxKind(kind.into()), text)
    }

    pub fn start_node(&mut self, kind: SyntaxKind) {
        self.inner.start_node(rowan::SyntaxKind(kind.into()))
    }

    pub fn finish_node(&mut self) {
        self.inner.finish_node()
    }

    pub fn error(&mut self, error: ParseError, text_pos: TextUnit) {
        let error = SyntaxError::new(SyntaxErrorKind::ParseError(error), text_pos);
        self.errors.push(error)
    }
}
