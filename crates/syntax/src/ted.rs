//! Primitive tree editor, ed for trees
#![allow(unused)]
use std::ops::RangeInclusive;

use crate::{SyntaxElement, SyntaxNode};

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
pub fn insert_all(position: Position, elements: Vec<SyntaxElement>) {
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
