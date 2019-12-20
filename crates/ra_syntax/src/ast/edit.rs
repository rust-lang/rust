//! This module contains functions for editing syntax trees. As the trees are
//! immutable, all function here return a fresh copy of the tree, instead of
//! doing an in-place modification.
use std::{iter, ops::RangeInclusive};

use arrayvec::ArrayVec;
use rustc_hash::FxHashMap;

use crate::{
    algo,
    ast::{
        self,
        make::{self, tokens},
        AstNode, TypeBoundsOwner,
    },
    AstToken, Direction, InsertPosition, SmolStr, SyntaxElement, SyntaxKind,
    SyntaxKind::{ATTR, COMMENT, WHITESPACE},
    SyntaxNode, SyntaxToken, T,
};

impl ast::BinExpr {
    #[must_use]
    pub fn replace_op(&self, op: SyntaxKind) -> Option<ast::BinExpr> {
        let op_node: SyntaxElement = self.op_details()?.0.into();
        let to_insert: Option<SyntaxElement> = Some(tokens::op(op).into());
        let replace_range = RangeInclusive::new(op_node.clone(), op_node);
        Some(replace_children(self, replace_range, to_insert.into_iter()))
    }
}

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

impl ast::ItemList {
    #[must_use]
    pub fn append_items(&self, items: impl Iterator<Item = ast::ImplItem>) -> ast::ItemList {
        let mut res = self.clone();
        if !self.syntax().text().contains_char('\n') {
            res = res.make_multiline();
        }
        items.for_each(|it| res = res.append_item(it));
        res
    }

    #[must_use]
    pub fn append_item(&self, item: ast::ImplItem) -> ast::ItemList {
        let (indent, position) = match self.impl_items().last() {
            Some(it) => (
                leading_indent(it.syntax()).unwrap_or_default().to_string(),
                InsertPosition::After(it.syntax().clone().into()),
            ),
            None => match self.l_curly() {
                Some(it) => (
                    "    ".to_string() + &leading_indent(self.syntax()).unwrap_or_default(),
                    InsertPosition::After(it),
                ),
                None => return self.clone(),
            },
        };
        let ws = tokens::WsBuilder::new(&format!("\n{}", indent));
        let to_insert: ArrayVec<[SyntaxElement; 2]> =
            [ws.ws().into(), item.syntax().clone().into()].into();
        insert_children(self, position, to_insert.into_iter())
    }

    fn l_curly(&self) -> Option<SyntaxElement> {
        self.syntax().children_with_tokens().find(|it| it.kind() == T!['{'])
    }

    fn make_multiline(&self) -> ast::ItemList {
        let l_curly = match self.syntax().children_with_tokens().find(|it| it.kind() == T!['{']) {
            Some(it) => it,
            None => return self.clone(),
        };
        let sibling = match l_curly.next_sibling_or_token() {
            Some(it) => it,
            None => return self.clone(),
        };
        let existing_ws = match sibling.as_token() {
            None => None,
            Some(tok) if tok.kind() != WHITESPACE => None,
            Some(ws) => {
                if ws.text().contains('\n') {
                    return self.clone();
                }
                Some(ws.clone())
            }
        };

        let indent = leading_indent(self.syntax()).unwrap_or_default();
        let ws = tokens::WsBuilder::new(&format!("\n{}", indent));
        let to_insert = iter::once(ws.ws().into());
        match existing_ws {
            None => insert_children(self, InsertPosition::After(l_curly), to_insert),
            Some(ws) => {
                replace_children(self, RangeInclusive::new(ws.clone().into(), ws.into()), to_insert)
            }
        }
    }
}

impl ast::RecordFieldList {
    #[must_use]
    pub fn append_field(&self, field: &ast::RecordField) -> ast::RecordFieldList {
        self.insert_field(InsertPosition::Last, field)
    }

    #[must_use]
    pub fn insert_field(
        &self,
        position: InsertPosition<&'_ ast::RecordField>,
        field: &ast::RecordField,
    ) -> ast::RecordFieldList {
        let is_multiline = self.syntax().text().contains_char('\n');
        let ws;
        let space = if is_multiline {
            ws = tokens::WsBuilder::new(&format!(
                "\n{}    ",
                leading_indent(self.syntax()).unwrap_or_default()
            ));
            ws.ws()
        } else {
            tokens::single_space()
        };

        let mut to_insert: ArrayVec<[SyntaxElement; 4]> = ArrayVec::new();
        to_insert.push(space.into());
        to_insert.push(field.syntax().clone().into());
        to_insert.push(tokens::comma().into());

        macro_rules! after_l_curly {
            () => {{
                let anchor = match self.l_curly() {
                    Some(it) => it,
                    None => return self.clone(),
                };
                InsertPosition::After(anchor)
            }};
        }

        macro_rules! after_field {
            ($anchor:expr) => {
                if let Some(comma) = $anchor
                    .syntax()
                    .siblings_with_tokens(Direction::Next)
                    .find(|it| it.kind() == T![,])
                {
                    InsertPosition::After(comma)
                } else {
                    to_insert.insert(0, tokens::comma().into());
                    InsertPosition::After($anchor.syntax().clone().into())
                }
            };
        };

        let position = match position {
            InsertPosition::First => after_l_curly!(),
            InsertPosition::Last => {
                if !is_multiline {
                    // don't insert comma before curly
                    to_insert.pop();
                }
                match self.fields().last() {
                    Some(it) => after_field!(it),
                    None => after_l_curly!(),
                }
            }
            InsertPosition::Before(anchor) => {
                InsertPosition::Before(anchor.syntax().clone().into())
            }
            InsertPosition::After(anchor) => after_field!(anchor),
        };

        insert_children(self, position, to_insert.iter().cloned())
    }

    fn l_curly(&self) -> Option<SyntaxElement> {
        self.syntax().children_with_tokens().find(|it| it.kind() == T!['{'])
    }
}

impl ast::TypeParam {
    #[must_use]
    pub fn remove_bounds(&self) -> ast::TypeParam {
        let colon = match self.colon_token() {
            Some(it) => it,
            None => return self.clone(),
        };
        let end = match self.type_bound_list() {
            Some(it) => it.syntax().clone().into(),
            None => colon.clone().into(),
        };
        replace_children(self, RangeInclusive::new(colon.into(), end), iter::empty())
    }
}

#[must_use]
pub fn strip_attrs_and_docs<N: ast::AttrsOwner>(node: &N) -> N {
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
pub fn replace_descendants<N: AstNode, D: AstNode>(
    parent: &N,
    replacement_map: impl Iterator<Item = (D, D)>,
) -> N {
    let map = replacement_map
        .map(|(from, to)| (from.syntax().clone().into(), to.syntax().clone().into()))
        .collect::<FxHashMap<_, _>>();
    let new_syntax = algo::replace_descendants(parent.syntax(), &map);
    N::cast(new_syntax).unwrap()
}

#[derive(Debug, Clone, Copy)]
pub struct IndentLevel(pub u8);

impl From<u8> for IndentLevel {
    fn from(level: u8) -> IndentLevel {
        IndentLevel(level)
    }
}

impl IndentLevel {
    pub fn from_node(node: &SyntaxNode) -> IndentLevel {
        let first_token = match node.first_token() {
            Some(it) => it,
            None => return IndentLevel(0),
        };
        for ws in prev_tokens(first_token).filter_map(ast::Whitespace::cast) {
            let text = ws.syntax().text();
            if let Some(pos) = text.rfind('\n') {
                let level = text[pos + 1..].chars().count() / 4;
                return IndentLevel(level as u8);
            }
        }
        IndentLevel(0)
    }

    pub fn increase_indent<N: AstNode>(self, node: N) -> N {
        N::cast(self._increase_indent(node.syntax().clone())).unwrap()
    }

    fn _increase_indent(self, node: SyntaxNode) -> SyntaxNode {
        let replacements: FxHashMap<SyntaxElement, SyntaxElement> = node
            .descendants_with_tokens()
            .filter_map(|el| el.into_token())
            .filter_map(ast::Whitespace::cast)
            .filter(|ws| {
                let text = ws.syntax().text();
                text.contains('\n')
            })
            .map(|ws| {
                (
                    ws.syntax().clone().into(),
                    make::tokens::whitespace(&format!(
                        "{}{:width$}",
                        ws.syntax().text(),
                        "",
                        width = self.0 as usize * 4
                    ))
                    .into(),
                )
            })
            .collect();
        algo::replace_descendants(&node, &replacements)
    }

    pub fn decrease_indent<N: AstNode>(self, node: N) -> N {
        N::cast(self._decrease_indent(node.syntax().clone())).unwrap()
    }

    fn _decrease_indent(self, node: SyntaxNode) -> SyntaxNode {
        let replacements: FxHashMap<SyntaxElement, SyntaxElement> = node
            .descendants_with_tokens()
            .filter_map(|el| el.into_token())
            .filter_map(ast::Whitespace::cast)
            .filter(|ws| {
                let text = ws.syntax().text();
                text.contains('\n')
            })
            .map(|ws| {
                (
                    ws.syntax().clone().into(),
                    make::tokens::whitespace(
                        &ws.syntax()
                            .text()
                            .replace(&format!("\n{:1$}", "", self.0 as usize * 4), "\n"),
                    )
                    .into(),
                )
            })
            .collect();
        algo::replace_descendants(&node, &replacements)
    }
}

// FIXME: replace usages with IndentLevel above
fn leading_indent(node: &SyntaxNode) -> Option<SmolStr> {
    for token in prev_tokens(node.first_token()?) {
        if let Some(ws) = ast::Whitespace::cast(token.clone()) {
            let ws_text = ws.text();
            if let Some(pos) = ws_text.rfind('\n') {
                return Some(ws_text[pos + 1..].into());
            }
        }
        if token.text().contains('\n') {
            break;
        }
    }
    None
}

fn prev_tokens(token: SyntaxToken) -> impl Iterator<Item = SyntaxToken> {
    iter::successors(Some(token), |token| token.prev_token())
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

#[test]
fn test_increase_indent() {
    let arm_list = {
        let arm = make::match_arm(iter::once(make::placeholder_pat().into()), make::expr_unit());
        make::match_arm_list(vec![arm.clone(), arm])
    };
    assert_eq!(
        arm_list.syntax().to_string(),
        "{
    _ => (),
    _ => (),
}"
    );
    let indented = IndentLevel(2).increase_indent(arm_list);
    assert_eq!(
        indented.syntax().to_string(),
        "{
            _ => (),
            _ => (),
        }"
    );
}
