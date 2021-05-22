//! This module contains functions for editing syntax trees. As the trees are
//! immutable, all function here return a fresh copy of the tree, instead of
//! doing an in-place modification.
use std::{
    fmt, iter,
    ops::{self, RangeInclusive},
};

use crate::{
    algo,
    ast::{self, make, AstNode},
    ted, AstToken, NodeOrToken, SyntaxElement, SyntaxKind,
    SyntaxKind::{ATTR, COMMENT, WHITESPACE},
    SyntaxNode, SyntaxToken,
};

impl ast::BinExpr {
    #[must_use]
    pub fn replace_op(&self, op: SyntaxKind) -> Option<ast::BinExpr> {
        let op_node: SyntaxElement = self.op_details()?.0.into();
        let to_insert: Option<SyntaxElement> = Some(make::token(op).into());
        Some(self.replace_children(single_node(op_node), to_insert))
    }
}

impl ast::UseTree {
    /// Splits off the given prefix, making it the path component of the use tree, appending the rest of the path to all UseTreeList items.
    #[must_use]
    pub fn split_prefix(&self, prefix: &ast::Path) -> ast::UseTree {
        let suffix = if self.path().as_ref() == Some(prefix) && self.use_tree_list().is_none() {
            make::path_unqualified(make::path_segment_self())
        } else {
            match split_path_prefix(&prefix) {
                Some(it) => it,
                None => return self.clone(),
            }
        };

        let use_tree = make::use_tree(
            suffix,
            self.use_tree_list(),
            self.rename(),
            self.star_token().is_some(),
        );
        let nested = make::use_tree_list(iter::once(use_tree));
        return make::use_tree(prefix.clone(), Some(nested), None, false);

        fn split_path_prefix(prefix: &ast::Path) -> Option<ast::Path> {
            let parent = prefix.parent_path()?;
            let segment = parent.segment()?;
            if algo::has_errors(segment.syntax()) {
                return None;
            }
            let mut res = make::path_unqualified(segment);
            for p in iter::successors(parent.parent_path(), |it| it.parent_path()) {
                res = make::path_qualified(res, p.segment()?);
            }
            Some(res)
        }
    }
}

#[must_use]
pub fn remove_attrs_and_docs<N: ast::AttrsOwner>(node: &N) -> N {
    N::cast(remove_attrs_and_docs_inner(node.syntax().clone())).unwrap()
}

fn remove_attrs_and_docs_inner(mut node: SyntaxNode) -> SyntaxNode {
    while let Some(start) =
        node.children_with_tokens().find(|it| it.kind() == ATTR || it.kind() == COMMENT)
    {
        let end = match &start.next_sibling_or_token() {
            Some(el) if el.kind() == WHITESPACE => el.clone(),
            Some(_) | None => start.clone(),
        };
        node = algo::replace_children(&node, start..=end, &mut iter::empty());
    }
    node
}

#[derive(Debug, Clone, Copy)]
pub struct IndentLevel(pub u8);

impl From<u8> for IndentLevel {
    fn from(level: u8) -> IndentLevel {
        IndentLevel(level)
    }
}

impl fmt::Display for IndentLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spaces = "                                        ";
        let buf;
        let len = self.0 as usize * 4;
        let indent = if len <= spaces.len() {
            &spaces[..len]
        } else {
            buf = iter::repeat(' ').take(len).collect::<String>();
            &buf
        };
        fmt::Display::fmt(indent, f)
    }
}

impl ops::Add<u8> for IndentLevel {
    type Output = IndentLevel;
    fn add(self, rhs: u8) -> IndentLevel {
        IndentLevel(self.0 + rhs)
    }
}

impl IndentLevel {
    pub fn single() -> IndentLevel {
        IndentLevel(0)
    }
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
    pub fn from_element(element: &SyntaxElement) -> IndentLevel {
        match element {
            rowan::NodeOrToken::Node(it) => IndentLevel::from_node(it),
            rowan::NodeOrToken::Token(it) => IndentLevel::from_token(it),
        }
    }

    pub fn from_node(node: &SyntaxNode) -> IndentLevel {
        match node.first_token() {
            Some(it) => Self::from_token(&it),
            None => IndentLevel(0),
        }
    }

    pub fn from_token(token: &SyntaxToken) -> IndentLevel {
        for ws in prev_tokens(token.clone()).filter_map(ast::Whitespace::cast) {
            let text = ws.syntax().text();
            if let Some(pos) = text.rfind('\n') {
                let level = text[pos + 1..].chars().count() / 4;
                return IndentLevel(level as u8);
            }
        }
        IndentLevel(0)
    }

    /// XXX: this intentionally doesn't change the indent of the very first token.
    /// Ie, in something like
    /// ```
    /// fn foo() {
    ///    92
    /// }
    /// ```
    /// if you indent the block, the `{` token would stay put.
    fn increase_indent(self, node: SyntaxNode) -> SyntaxNode {
        let res = node.clone_subtree().clone_for_update();
        let tokens = res.preorder_with_tokens().filter_map(|event| match event {
            rowan::WalkEvent::Leave(NodeOrToken::Token(it)) => Some(it),
            _ => None,
        });
        for token in tokens {
            if let Some(ws) = ast::Whitespace::cast(token) {
                if ws.text().contains('\n') {
                    let new_ws = make::tokens::whitespace(&format!("{}{}", ws.syntax(), self));
                    ted::replace(ws.syntax(), &new_ws)
                }
            }
        }
        res.clone_subtree()
    }

    fn decrease_indent(self, node: SyntaxNode) -> SyntaxNode {
        let res = node.clone_subtree().clone_for_update();
        let tokens = res.preorder_with_tokens().filter_map(|event| match event {
            rowan::WalkEvent::Leave(NodeOrToken::Token(it)) => Some(it),
            _ => None,
        });
        for token in tokens {
            if let Some(ws) = ast::Whitespace::cast(token) {
                if ws.text().contains('\n') {
                    let new_ws = make::tokens::whitespace(
                        &ws.syntax().text().replace(&format!("\n{}", self), "\n"),
                    );
                    ted::replace(ws.syntax(), &new_ws)
                }
            }
        }
        res.clone_subtree()
    }
}

fn prev_tokens(token: SyntaxToken) -> impl Iterator<Item = SyntaxToken> {
    iter::successors(Some(token), |token| token.prev_token())
}

pub trait AstNodeEdit: AstNode + Clone + Sized {
    #[must_use]
    fn replace_children(
        &self,
        to_replace: RangeInclusive<SyntaxElement>,
        to_insert: impl IntoIterator<Item = SyntaxElement>,
    ) -> Self {
        let new_syntax = algo::replace_children(self.syntax(), to_replace, to_insert);
        Self::cast(new_syntax).unwrap()
    }
    fn indent_level(&self) -> IndentLevel {
        IndentLevel::from_node(self.syntax())
    }
    #[must_use]
    fn indent(&self, level: IndentLevel) -> Self {
        Self::cast(level.increase_indent(self.syntax().clone())).unwrap()
    }
    #[must_use]
    fn dedent(&self, level: IndentLevel) -> Self {
        Self::cast(level.decrease_indent(self.syntax().clone())).unwrap()
    }
    #[must_use]
    fn reset_indent(&self) -> Self {
        let level = IndentLevel::from_node(self.syntax());
        self.dedent(level)
    }
}

impl<N: AstNode + Clone> AstNodeEdit for N {}

fn single_node(element: impl Into<SyntaxElement>) -> RangeInclusive<SyntaxElement> {
    let element = element.into();
    element.clone()..=element
}

#[test]
fn test_increase_indent() {
    let arm_list = {
        let arm = make::match_arm(iter::once(make::wildcard_pat().into()), make::expr_unit());
        make::match_arm_list(vec![arm.clone(), arm])
    };
    assert_eq!(
        arm_list.syntax().to_string(),
        "{
    _ => (),
    _ => (),
}"
    );
    let indented = arm_list.indent(IndentLevel(2));
    assert_eq!(
        indented.syntax().to_string(),
        "{
            _ => (),
            _ => (),
        }"
    );
}
