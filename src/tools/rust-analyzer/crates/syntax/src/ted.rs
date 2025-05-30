//! Primitive tree editor, ed for trees.
//!
//! The `_raw`-suffixed functions insert elements as is, unsuffixed versions fix
//! up elements around the edges.
use std::{mem, ops::RangeInclusive};

use parser::T;
use rowan::TextSize;

use crate::{
    SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken,
    ast::{self, AstNode, edit::IndentLevel, make},
};

/// Utility trait to allow calling `ted` functions with references or owned
/// nodes. Do not use outside of this module.
pub trait Element {
    fn syntax_element(self) -> SyntaxElement;
}

impl<E: Element + Clone> Element for &'_ E {
    fn syntax_element(self) -> SyntaxElement {
        self.clone().syntax_element()
    }
}
impl Element for SyntaxElement {
    fn syntax_element(self) -> SyntaxElement {
        self
    }
}
impl Element for SyntaxNode {
    fn syntax_element(self) -> SyntaxElement {
        self.into()
    }
}
impl Element for SyntaxToken {
    fn syntax_element(self) -> SyntaxElement {
        self.into()
    }
}

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
    pub fn after(elem: impl Element) -> Position {
        let repr = PositionRepr::After(elem.syntax_element());
        Position { repr }
    }
    pub fn before(elem: impl Element) -> Position {
        let elem = elem.syntax_element();
        let repr = match elem.prev_sibling_or_token() {
            Some(it) => PositionRepr::After(it),
            None => PositionRepr::FirstChild(elem.parent().unwrap()),
        };
        Position { repr }
    }
    pub fn first_child_of(node: &(impl Into<SyntaxNode> + Clone)) -> Position {
        let repr = PositionRepr::FirstChild(node.clone().into());
        Position { repr }
    }
    pub fn last_child_of(node: &(impl Into<SyntaxNode> + Clone)) -> Position {
        let node = node.clone().into();
        let repr = match node.last_child_or_token() {
            Some(it) => PositionRepr::After(it),
            None => PositionRepr::FirstChild(node),
        };
        Position { repr }
    }
    pub fn offset(&self) -> TextSize {
        match &self.repr {
            PositionRepr::FirstChild(node) => node.text_range().start(),
            PositionRepr::After(elem) => elem.text_range().end(),
        }
    }
}

pub fn insert(position: Position, elem: impl Element) {
    insert_all(position, vec![elem.syntax_element()]);
}
pub fn insert_raw(position: Position, elem: impl Element) {
    insert_all_raw(position, vec![elem.syntax_element()]);
}
pub fn insert_all(position: Position, mut elements: Vec<SyntaxElement>) {
    if let Some(first) = elements.first() {
        if let Some(ws) = ws_before(&position, first) {
            elements.insert(0, ws.into());
        }
    }
    if let Some(last) = elements.last() {
        if let Some(ws) = ws_after(&position, last) {
            elements.push(ws.into());
        }
    }
    insert_all_raw(position, elements);
}
pub fn insert_all_raw(position: Position, elements: Vec<SyntaxElement>) {
    let (parent, index) = match position.repr {
        PositionRepr::FirstChild(parent) => (parent, 0),
        PositionRepr::After(child) => (child.parent().unwrap(), child.index() + 1),
    };
    parent.splice_children(index..index, elements);
}

pub fn remove(elem: impl Element) {
    elem.syntax_element().detach();
}
pub fn remove_all(range: RangeInclusive<SyntaxElement>) {
    replace_all(range, Vec::new());
}
pub fn remove_all_iter(range: impl IntoIterator<Item = SyntaxElement>) {
    let mut it = range.into_iter();
    if let Some(mut first) = it.next() {
        match it.last() {
            Some(mut last) => {
                if first.index() > last.index() {
                    mem::swap(&mut first, &mut last);
                }
                remove_all(first..=last);
            }
            None => remove(first),
        }
    }
}

pub fn replace(old: impl Element, new: impl Element) {
    replace_with_many(old, vec![new.syntax_element()]);
}
pub fn replace_with_many(old: impl Element, new: Vec<SyntaxElement>) {
    let old = old.syntax_element();
    replace_all(old.clone()..=old, new);
}
pub fn replace_all(range: RangeInclusive<SyntaxElement>, new: Vec<SyntaxElement>) {
    let start = range.start().index();
    let end = range.end().index();
    let parent = range.start().parent().unwrap();
    parent.splice_children(start..end + 1, new);
}

pub fn append_child(node: &(impl Into<SyntaxNode> + Clone), child: impl Element) {
    let position = Position::last_child_of(node);
    insert(position, child);
}
pub fn append_child_raw(node: &(impl Into<SyntaxNode> + Clone), child: impl Element) {
    let position = Position::last_child_of(node);
    insert_raw(position, child);
}

pub fn prepend_child(node: &(impl Into<SyntaxNode> + Clone), child: impl Element) {
    let position = Position::first_child_of(node);
    insert(position, child);
}

fn ws_before(position: &Position, new: &SyntaxElement) -> Option<SyntaxToken> {
    let prev = match &position.repr {
        PositionRepr::FirstChild(_) => return None,
        PositionRepr::After(it) => it,
    };

    if prev.kind() == T!['{'] && new.kind() == SyntaxKind::USE {
        if let Some(item_list) = prev.parent().and_then(ast::ItemList::cast) {
            let mut indent = IndentLevel::from_element(&item_list.syntax().clone().into());
            indent.0 += 1;
            return Some(make::tokens::whitespace(&format!("\n{indent}")));
        }
    }

    if prev.kind() == T!['{'] && ast::Stmt::can_cast(new.kind()) {
        if let Some(stmt_list) = prev.parent().and_then(ast::StmtList::cast) {
            let mut indent = IndentLevel::from_element(&stmt_list.syntax().clone().into());
            indent.0 += 1;
            return Some(make::tokens::whitespace(&format!("\n{indent}")));
        }
    }

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
    if left.kind() == T![<] || right.kind() == T![>] {
        return None;
    }
    if left.kind() == T![&] && right.kind() == SyntaxKind::LIFETIME {
        return None;
    }
    if right.kind() == SyntaxKind::GENERIC_ARG_LIST {
        return None;
    }

    if right.kind() == SyntaxKind::USE {
        let mut indent = IndentLevel::from_element(left);
        if left.kind() == SyntaxKind::USE {
            indent.0 = IndentLevel::from_element(right).0.max(indent.0);
        }
        return Some(make::tokens::whitespace(&format!("\n{indent}")));
    }
    if left.kind() == SyntaxKind::ATTR {
        let mut indent = IndentLevel::from_element(right);
        if right.kind() == SyntaxKind::ATTR {
            indent.0 = IndentLevel::from_element(left).0.max(indent.0);
        }
        return Some(make::tokens::whitespace(&format!("\n{indent}")));
    }
    Some(make::tokens::single_space())
}
