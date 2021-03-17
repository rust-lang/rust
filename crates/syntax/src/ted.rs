//! Primitive tree editor, ed for trees.
//!
//! The `_raw`-suffixed functions insert elements as is, unsuffixed versions fix
//! up elements around the edges.
use std::ops::RangeInclusive;

use parser::T;

use crate::{ast::make, SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken};

#[derive(Debug)]
pub struct Position {
    repr: PositionRepr,
}

#[derive(Debug)]
enum PositionRepr {
    FirstChild(SyntaxNode),
    After(SyntaxElement),
}

impl Position {
    pub fn after(elem: impl Into<SyntaxElement>) -> Position {
        let repr = PositionRepr::After(elem.into());
        Position { repr }
    }
    pub fn before(elem: impl Into<SyntaxElement>) -> Position {
        let elem = elem.into();
        let repr = match elem.prev_sibling_or_token() {
            Some(it) => PositionRepr::After(it),
            None => PositionRepr::FirstChild(elem.parent().unwrap()),
        };
        Position { repr }
    }
    pub fn first_child_of(node: impl Into<SyntaxNode>) -> Position {
        let repr = PositionRepr::FirstChild(node.into());
        Position { repr }
    }
    pub fn last_child_of(node: impl Into<SyntaxNode>) -> Position {
        let node = node.into();
        let repr = match node.last_child_or_token() {
            Some(it) => PositionRepr::After(it),
            None => PositionRepr::FirstChild(node),
        };
        Position { repr }
    }
}

pub fn insert(position: Position, elem: impl Into<SyntaxElement>) {
    insert_all(position, vec![elem.into()])
}
pub fn insert_raw(position: Position, elem: impl Into<SyntaxElement>) {
    insert_all_raw(position, vec![elem.into()])
}
pub fn insert_all(position: Position, mut elements: Vec<SyntaxElement>) {
    if let Some(first) = elements.first() {
        if let Some(ws) = ws_before(&position, first) {
            elements.insert(0, ws.into())
        }
    }
    if let Some(last) = elements.last() {
        if let Some(ws) = ws_after(&position, last) {
            elements.push(ws.into())
        }
    }
    insert_all_raw(position, elements)
}
pub fn insert_all_raw(position: Position, elements: Vec<SyntaxElement>) {
    let (parent, index) = match position.repr {
        PositionRepr::FirstChild(parent) => (parent, 0),
        PositionRepr::After(child) => (child.parent().unwrap(), child.index() + 1),
    };
    parent.splice_children(index..index, elements);
}

pub fn remove(elem: impl Into<SyntaxElement>) {
    let elem = elem.into();
    remove_all(elem.clone()..=elem)
}
pub fn remove_all(range: RangeInclusive<SyntaxElement>) {
    replace_all(range, Vec::new())
}

pub fn replace(old: impl Into<SyntaxElement>, new: impl Into<SyntaxElement>) {
    let old = old.into();
    replace_all(old.clone()..=old, vec![new.into()])
}
pub fn replace_all(range: RangeInclusive<SyntaxElement>, new: Vec<SyntaxElement>) {
    let start = range.start().index();
    let end = range.end().index();
    let parent = range.start().parent().unwrap();
    parent.splice_children(start..end + 1, new)
}

pub fn append_child(node: impl Into<SyntaxNode>, child: impl Into<SyntaxElement>) {
    let position = Position::last_child_of(node);
    insert(position, child)
}
pub fn append_child_raw(node: impl Into<SyntaxNode>, child: impl Into<SyntaxElement>) {
    let position = Position::last_child_of(node);
    insert_raw(position, child)
}

fn ws_before(position: &Position, new: &SyntaxElement) -> Option<SyntaxToken> {
    let prev = match &position.repr {
        PositionRepr::FirstChild(_) => return None,
        PositionRepr::After(it) => it,
    };
    ws_between(prev, new)
}
fn ws_after(position: &Position, new: &SyntaxElement) -> Option<SyntaxToken> {
    let next = match &position.repr {
        PositionRepr::FirstChild(parent) => parent.first_child_or_token()?,
        PositionRepr::After(sibling) => sibling.next_sibling_or_token()?,
    };
    ws_between(new, &next)
}
fn ws_between(left: &SyntaxElement, right: &SyntaxElement) -> Option<SyntaxToken> {
    if left.kind() == SyntaxKind::WHITESPACE || right.kind() == SyntaxKind::WHITESPACE {
        return None;
    }
    if right.kind() == T![;] || right.kind() == T![,] {
        return None;
    }
    Some(make::tokens::single_space())
}
