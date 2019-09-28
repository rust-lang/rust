//! This module contains functions for editing syntax trees. As the trees are
//! immutable, all function here return a fresh copy of the tree, instead of
//! doing an in-place modification.
use std::{iter, ops::RangeInclusive};

use arrayvec::ArrayVec;

use crate::{
    algo,
    ast::{self, make, AstNode},
    InsertPosition, SyntaxElement,
    SyntaxKind::{ATTR, COMMENT, WHITESPACE},
    SyntaxNode,
};

impl ast::FnDef {
    #[must_use]
    pub fn with_body(&self, body: ast::Block) -> ast::FnDef {
        let mut to_insert: ArrayVec<[SyntaxElement; 2]> = ArrayVec::new();
        let old_body_or_semi: SyntaxElement = if let Some(old_body) = self.body() {
            old_body.syntax().clone().into()
        } else if let Some(semi) = self.semicolon_token() {
            to_insert.push(make::tokens::single_space().into());
            semi.into()
        } else {
            to_insert.push(make::tokens::single_space().into());
            to_insert.push(body.syntax().clone().into());
            return insert_children(self, InsertPosition::Last, to_insert.into_iter());
        };
        to_insert.push(body.syntax().clone().into());
        let replace_range = RangeInclusive::new(old_body_or_semi.clone(), old_body_or_semi);
        replace_children(self, replace_range, to_insert.into_iter())
    }
}

pub fn strip_attrs_and_docs<N: ast::AttrsOwner>(node: N) -> N {
    N::cast(strip_attrs_and_docs_inner(node.syntax().clone())).unwrap()
}

fn strip_attrs_and_docs_inner(mut node: SyntaxNode) -> SyntaxNode {
    while let Some(start) =
        node.children_with_tokens().find(|it| it.kind() == ATTR || it.kind() == COMMENT)
    {
        let end = match &start.next_sibling_or_token() {
            Some(el) if el.kind() == WHITESPACE => el.clone(),
            Some(_) | None => start.clone(),
        };
        node = algo::replace_children(&node, RangeInclusive::new(start, end), &mut iter::empty());
    }
    node
}

#[must_use]
fn insert_children<N: AstNode>(
    parent: &N,
    position: InsertPosition<SyntaxElement>,
    mut to_insert: impl Iterator<Item = SyntaxElement>,
) -> N {
    let new_syntax = algo::insert_children(parent.syntax(), position, &mut to_insert);
    N::cast(new_syntax).unwrap()
}

#[must_use]
fn replace_children<N: AstNode>(
    parent: &N,
    to_replace: RangeInclusive<SyntaxElement>,
    mut to_insert: impl Iterator<Item = SyntaxElement>,
) -> N {
    let new_syntax = algo::replace_children(parent.syntax(), to_replace, &mut to_insert);
    N::cast(new_syntax).unwrap()
}
