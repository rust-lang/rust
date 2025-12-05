//! This module contains functions for editing syntax trees. As the trees are
//! immutable, all function here return a fresh copy of the tree, instead of
//! doing an in-place modification.
use std::{fmt, iter, ops};

use crate::{
    AstToken, NodeOrToken, SyntaxElement, SyntaxNode, SyntaxToken,
    ast::{self, AstNode, make},
    syntax_editor::{SyntaxEditor, SyntaxMappingBuilder},
    ted,
};

use super::syntax_factory::SyntaxFactory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            buf = " ".repeat(len);
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
    /// For example, in something like:
    /// ```
    /// fn foo() -> i32 {
    ///    92
    /// }
    /// ```
    /// if you indent the block, the `{` token would stay put.
    pub(super) fn increase_indent(self, node: &SyntaxNode) {
        let tokens = node.preorder_with_tokens().filter_map(|event| match event {
            rowan::WalkEvent::Leave(NodeOrToken::Token(it)) => Some(it),
            _ => None,
        });
        for token in tokens {
            if let Some(ws) = ast::Whitespace::cast(token)
                && ws.text().contains('\n')
            {
                let new_ws = make::tokens::whitespace(&format!("{}{self}", ws.syntax()));
                ted::replace(ws.syntax(), &new_ws);
            }
        }
    }

    pub(super) fn clone_increase_indent(self, node: &SyntaxNode) -> SyntaxNode {
        let node = node.clone_subtree();
        let mut editor = SyntaxEditor::new(node.clone());
        let tokens = node
            .preorder_with_tokens()
            .filter_map(|event| match event {
                rowan::WalkEvent::Leave(NodeOrToken::Token(it)) => Some(it),
                _ => None,
            })
            .filter_map(ast::Whitespace::cast)
            .filter(|ws| ws.text().contains('\n'));
        for ws in tokens {
            let new_ws = make::tokens::whitespace(&format!("{}{self}", ws.syntax()));
            editor.replace(ws.syntax(), &new_ws);
        }
        editor.finish().new_root().clone()
    }

    pub(super) fn decrease_indent(self, node: &SyntaxNode) {
        let tokens = node.preorder_with_tokens().filter_map(|event| match event {
            rowan::WalkEvent::Leave(NodeOrToken::Token(it)) => Some(it),
            _ => None,
        });
        for token in tokens {
            if let Some(ws) = ast::Whitespace::cast(token)
                && ws.text().contains('\n')
            {
                let new_ws = make::tokens::whitespace(
                    &ws.syntax().text().replace(&format!("\n{self}"), "\n"),
                );
                ted::replace(ws.syntax(), &new_ws);
            }
        }
    }

    pub(super) fn clone_decrease_indent(self, node: &SyntaxNode) -> SyntaxNode {
        let node = node.clone_subtree();
        let mut editor = SyntaxEditor::new(node.clone());
        let tokens = node
            .preorder_with_tokens()
            .filter_map(|event| match event {
                rowan::WalkEvent::Leave(NodeOrToken::Token(it)) => Some(it),
                _ => None,
            })
            .filter_map(ast::Whitespace::cast)
            .filter(|ws| ws.text().contains('\n'));
        for ws in tokens {
            let new_ws =
                make::tokens::whitespace(&ws.syntax().text().replace(&format!("\n{self}"), "\n"));
            editor.replace(ws.syntax(), &new_ws);
        }
        editor.finish().new_root().clone()
    }
}

fn prev_tokens(token: SyntaxToken) -> impl Iterator<Item = SyntaxToken> {
    iter::successors(Some(token), |token| token.prev_token())
}

pub trait AstNodeEdit: AstNode + Clone + Sized {
    fn indent_level(&self) -> IndentLevel {
        IndentLevel::from_node(self.syntax())
    }
    #[must_use]
    fn indent(&self, level: IndentLevel) -> Self {
        Self::cast(level.clone_increase_indent(self.syntax())).unwrap()
    }
    #[must_use]
    fn indent_with_mapping(&self, level: IndentLevel, make: &SyntaxFactory) -> Self {
        let new_node = self.indent(level);
        if let Some(mut mapping) = make.mappings() {
            let mut builder = SyntaxMappingBuilder::new(new_node.syntax().clone());
            for (old, new) in self.syntax().children().zip(new_node.syntax().children()) {
                builder.map_node(old, new);
            }
            builder.finish(&mut mapping);
        }
        new_node
    }
    #[must_use]
    fn dedent(&self, level: IndentLevel) -> Self {
        Self::cast(level.clone_decrease_indent(self.syntax())).unwrap()
    }
    #[must_use]
    fn reset_indent(&self) -> Self {
        let level = IndentLevel::from_node(self.syntax());
        self.dedent(level)
    }
}

impl<N: AstNode + Clone> AstNodeEdit for N {}

#[test]
fn test_increase_indent() {
    let arm_list = {
        let arm = make::match_arm(make::wildcard_pat().into(), None, make::ext::expr_unit());
        make::match_arm_list([arm.clone(), arm])
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
