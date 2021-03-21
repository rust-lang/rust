//! This module contains functions for editing syntax trees. As the trees are
//! immutable, all function here return a fresh copy of the tree, instead of
//! doing an in-place modification.
use std::{
    fmt, iter,
    ops::{self, RangeInclusive},
};

use arrayvec::ArrayVec;

use crate::{
    algo::{self, neighbor, SyntaxRewriter},
    ast::{
        self,
        make::{self, tokens},
        AstNode, GenericParamsOwner, NameOwner, TypeBoundsOwner,
    },
    AstToken, Direction, InsertPosition, SmolStr, SyntaxElement, SyntaxKind,
    SyntaxKind::{ATTR, COMMENT, WHITESPACE},
    SyntaxNode, SyntaxToken, T,
};

impl ast::BinExpr {
    #[must_use]
    pub fn replace_op(&self, op: SyntaxKind) -> Option<ast::BinExpr> {
        let op_node: SyntaxElement = self.op_details()?.0.into();
        let to_insert: Option<SyntaxElement> = Some(make::token(op).into());
        Some(self.replace_children(single_node(op_node), to_insert))
    }
}

impl ast::Fn {
    #[must_use]
    pub fn with_body(&self, body: ast::BlockExpr) -> ast::Fn {
        let mut to_insert: ArrayVec<[SyntaxElement; 2]> = ArrayVec::new();
        let old_body_or_semi: SyntaxElement = if let Some(old_body) = self.body() {
            old_body.syntax().clone().into()
        } else if let Some(semi) = self.semicolon_token() {
            to_insert.push(make::tokens::single_space().into());
            semi.into()
        } else {
            to_insert.push(make::tokens::single_space().into());
            to_insert.push(body.syntax().clone().into());
            return self.insert_children(InsertPosition::Last, to_insert);
        };
        to_insert.push(body.syntax().clone().into());
        self.replace_children(single_node(old_body_or_semi), to_insert)
    }

    #[must_use]
    pub fn with_generic_param_list(&self, generic_args: ast::GenericParamList) -> ast::Fn {
        if let Some(old) = self.generic_param_list() {
            return self.replace_descendant(old, generic_args);
        }

        let anchor = self.name().expect("The function must have a name").syntax().clone();

        let mut to_insert: ArrayVec<[SyntaxElement; 1]> = ArrayVec::new();
        to_insert.push(generic_args.syntax().clone().into());
        self.insert_children(InsertPosition::After(anchor.into()), to_insert)
    }
}

fn make_multiline<N>(node: N) -> N
where
    N: AstNode + Clone,
{
    let l_curly = match node.syntax().children_with_tokens().find(|it| it.kind() == T!['{']) {
        Some(it) => it,
        None => return node,
    };
    let sibling = match l_curly.next_sibling_or_token() {
        Some(it) => it,
        None => return node,
    };
    let existing_ws = match sibling.as_token() {
        None => None,
        Some(tok) if tok.kind() != WHITESPACE => None,
        Some(ws) => {
            if ws.text().contains('\n') {
                return node;
            }
            Some(ws.clone())
        }
    };

    let indent = leading_indent(node.syntax()).unwrap_or_default();
    let ws = tokens::WsBuilder::new(&format!("\n{}", indent));
    let to_insert = iter::once(ws.ws().into());
    match existing_ws {
        None => node.insert_children(InsertPosition::After(l_curly), to_insert),
        Some(ws) => node.replace_children(single_node(ws), to_insert),
    }
}

impl ast::Impl {
    #[must_use]
    pub fn with_assoc_item_list(&self, items: ast::AssocItemList) -> ast::Impl {
        let mut to_insert: ArrayVec<[SyntaxElement; 2]> = ArrayVec::new();
        if let Some(old_items) = self.assoc_item_list() {
            let to_replace: SyntaxElement = old_items.syntax().clone().into();
            to_insert.push(items.syntax().clone().into());
            self.replace_children(single_node(to_replace), to_insert)
        } else {
            to_insert.push(make::tokens::single_space().into());
            to_insert.push(items.syntax().clone().into());
            self.insert_children(InsertPosition::Last, to_insert)
        }
    }
}

impl ast::AssocItemList {
    #[must_use]
    pub fn append_items(
        &self,
        items: impl IntoIterator<Item = ast::AssocItem>,
    ) -> ast::AssocItemList {
        let mut res = self.clone();
        if !self.syntax().text().contains_char('\n') {
            res = make_multiline(res);
        }
        items.into_iter().for_each(|it| res = res.append_item(it));
        res.fixup_trailing_whitespace().unwrap_or(res)
    }

    #[must_use]
    pub fn append_item(&self, item: ast::AssocItem) -> ast::AssocItemList {
        let (indent, position, whitespace) = match self.assoc_items().last() {
            Some(it) => (
                leading_indent(it.syntax()).unwrap_or_default().to_string(),
                InsertPosition::After(it.syntax().clone().into()),
                "\n\n",
            ),
            None => match self.l_curly_token() {
                Some(it) => (
                    "    ".to_string() + &leading_indent(self.syntax()).unwrap_or_default(),
                    InsertPosition::After(it.into()),
                    "\n",
                ),
                None => return self.clone(),
            },
        };
        let ws = tokens::WsBuilder::new(&format!("{}{}", whitespace, indent));
        let to_insert: ArrayVec<[SyntaxElement; 2]> =
            [ws.ws().into(), item.syntax().clone().into()].into();
        self.insert_children(position, to_insert)
    }

    /// Remove extra whitespace between last item and closing curly brace.
    fn fixup_trailing_whitespace(&self) -> Option<ast::AssocItemList> {
        let first_token_after_items =
            self.assoc_items().last()?.syntax().next_sibling_or_token()?;
        let last_token_before_curly = self.r_curly_token()?.prev_sibling_or_token()?;
        if last_token_before_curly != first_token_after_items {
            // there is something more between last item and
            // right curly than just whitespace - bail out
            return None;
        }
        let whitespace =
            last_token_before_curly.clone().into_token().and_then(ast::Whitespace::cast)?;
        let text = whitespace.syntax().text();
        let newline = text.rfind('\n')?;
        let keep = tokens::WsBuilder::new(&text[newline..]);
        Some(self.replace_children(
            first_token_after_items..=last_token_before_curly,
            std::iter::once(keep.ws().into()),
        ))
    }
}

impl ast::RecordExprFieldList {
    #[must_use]
    pub fn append_field(&self, field: &ast::RecordExprField) -> ast::RecordExprFieldList {
        self.insert_field(InsertPosition::Last, field)
    }

    #[must_use]
    pub fn insert_field(
        &self,
        position: InsertPosition<&'_ ast::RecordExprField>,
        field: &ast::RecordExprField,
    ) -> ast::RecordExprFieldList {
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
        to_insert.push(make::token(T![,]).into());

        macro_rules! after_l_curly {
            () => {{
                let anchor = match self.l_curly_token() {
                    Some(it) => it.into(),
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
                    to_insert.insert(0, make::token(T![,]).into());
                    InsertPosition::After($anchor.syntax().clone().into())
                }
            };
        }

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

        self.insert_children(position, to_insert)
    }
}

impl ast::TypeAlias {
    #[must_use]
    pub fn remove_bounds(&self) -> ast::TypeAlias {
        let colon = match self.colon_token() {
            Some(it) => it,
            None => return self.clone(),
        };
        let end = match self.type_bound_list() {
            Some(it) => it.syntax().clone().into(),
            None => colon.clone().into(),
        };
        self.replace_children(colon.into()..=end, iter::empty())
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
        self.replace_children(colon.into()..=end, iter::empty())
    }
}

impl ast::Path {
    #[must_use]
    pub fn with_segment(&self, segment: ast::PathSegment) -> ast::Path {
        if let Some(old) = self.segment() {
            return self.replace_children(
                single_node(old.syntax().clone()),
                iter::once(segment.syntax().clone().into()),
            );
        }
        self.clone()
    }
}

impl ast::PathSegment {
    #[must_use]
    pub fn with_generic_args(&self, type_args: ast::GenericArgList) -> ast::PathSegment {
        self._with_generic_args(type_args, false)
    }

    #[must_use]
    pub fn with_turbo_fish(&self, type_args: ast::GenericArgList) -> ast::PathSegment {
        self._with_generic_args(type_args, true)
    }

    fn _with_generic_args(&self, type_args: ast::GenericArgList, turbo: bool) -> ast::PathSegment {
        if let Some(old) = self.generic_arg_list() {
            return self.replace_children(
                single_node(old.syntax().clone()),
                iter::once(type_args.syntax().clone().into()),
            );
        }
        let mut to_insert: ArrayVec<[SyntaxElement; 2]> = ArrayVec::new();
        if turbo {
            to_insert.push(make::token(T![::]).into());
        }
        to_insert.push(type_args.syntax().clone().into());
        self.insert_children(InsertPosition::Last, to_insert)
    }
}

impl ast::Use {
    #[must_use]
    pub fn with_use_tree(&self, use_tree: ast::UseTree) -> ast::Use {
        if let Some(old) = self.use_tree() {
            return self.replace_descendant(old, use_tree);
        }
        self.clone()
    }

    pub fn remove(&self) -> SyntaxRewriter<'static> {
        let mut res = SyntaxRewriter::default();
        res.delete(self.syntax());
        let next_ws = self
            .syntax()
            .next_sibling_or_token()
            .and_then(|it| it.into_token())
            .and_then(ast::Whitespace::cast);
        if let Some(next_ws) = next_ws {
            let ws_text = next_ws.syntax().text();
            if let Some(rest) = ws_text.strip_prefix('\n') {
                if rest.is_empty() {
                    res.delete(next_ws.syntax())
                } else {
                    res.replace(next_ws.syntax(), &make::tokens::whitespace(rest));
                }
            }
        }
        res
    }
}

impl ast::UseTree {
    #[must_use]
    pub fn with_path(&self, path: ast::Path) -> ast::UseTree {
        if let Some(old) = self.path() {
            return self.replace_descendant(old, path);
        }
        self.clone()
    }

    #[must_use]
    pub fn with_use_tree_list(&self, use_tree_list: ast::UseTreeList) -> ast::UseTree {
        if let Some(old) = self.use_tree_list() {
            return self.replace_descendant(old, use_tree_list);
        }
        self.clone()
    }

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

    pub fn remove(&self) -> SyntaxRewriter<'static> {
        let mut res = SyntaxRewriter::default();
        res.delete(self.syntax());
        for &dir in [Direction::Next, Direction::Prev].iter() {
            if let Some(nb) = neighbor(self, dir) {
                self.syntax()
                    .siblings_with_tokens(dir)
                    .skip(1)
                    .take_while(|it| it.as_node() != Some(nb.syntax()))
                    .for_each(|el| res.delete(&el));
                return res;
            }
        }
        res
    }
}

impl ast::MatchArmList {
    #[must_use]
    pub fn append_arms(&self, items: impl IntoIterator<Item = ast::MatchArm>) -> ast::MatchArmList {
        let mut res = self.clone();
        res = res.strip_if_only_whitespace();
        if !res.syntax().text().contains_char('\n') {
            res = make_multiline(res);
        }
        items.into_iter().for_each(|it| res = res.append_arm(it));
        res
    }

    fn strip_if_only_whitespace(&self) -> ast::MatchArmList {
        let mut iter = self.syntax().children_with_tokens().skip_while(|it| it.kind() != T!['{']);
        iter.next(); // Eat the curly
        let mut inner = iter.take_while(|it| it.kind() != T!['}']);
        if !inner.clone().all(|it| it.kind() == WHITESPACE) {
            return self.clone();
        }
        let start = match inner.next() {
            Some(s) => s,
            None => return self.clone(),
        };
        let end = match inner.last() {
            Some(s) => s,
            None => start.clone(),
        };
        self.replace_children(start..=end, &mut iter::empty())
    }

    #[must_use]
    pub fn remove_placeholder(&self) -> ast::MatchArmList {
        let placeholder =
            self.arms().find(|arm| matches!(arm.pat(), Some(ast::Pat::WildcardPat(_))));
        if let Some(placeholder) = placeholder {
            self.remove_arm(&placeholder)
        } else {
            self.clone()
        }
    }

    #[must_use]
    fn remove_arm(&self, arm: &ast::MatchArm) -> ast::MatchArmList {
        let start = arm.syntax().clone();
        let end = if let Some(comma) = start
            .siblings_with_tokens(Direction::Next)
            .skip(1)
            .find(|it| !it.kind().is_trivia())
            .filter(|it| it.kind() == T![,])
        {
            comma
        } else {
            start.clone().into()
        };
        self.replace_children(start.into()..=end, None)
    }

    #[must_use]
    pub fn append_arm(&self, item: ast::MatchArm) -> ast::MatchArmList {
        let r_curly = match self.syntax().children_with_tokens().find(|it| it.kind() == T!['}']) {
            Some(t) => t,
            None => return self.clone(),
        };
        let position = InsertPosition::Before(r_curly);
        let arm_ws = tokens::WsBuilder::new("    ");
        let match_indent = &leading_indent(self.syntax()).unwrap_or_default();
        let match_ws = tokens::WsBuilder::new(&format!("\n{}", match_indent));
        let to_insert: ArrayVec<[SyntaxElement; 3]> =
            [arm_ws.ws().into(), item.syntax().clone().into(), match_ws.ws().into()].into();
        self.insert_children(position, to_insert)
    }
}

impl ast::GenericParamList {
    #[must_use]
    pub fn append_params(
        &self,
        params: impl IntoIterator<Item = ast::GenericParam>,
    ) -> ast::GenericParamList {
        let mut res = self.clone();
        params.into_iter().for_each(|it| res = res.append_param(it));
        res
    }

    #[must_use]
    pub fn append_param(&self, item: ast::GenericParam) -> ast::GenericParamList {
        let space = tokens::single_space();

        let mut to_insert: ArrayVec<[SyntaxElement; 4]> = ArrayVec::new();
        if self.generic_params().next().is_some() {
            to_insert.push(space.into());
        }
        to_insert.push(item.syntax().clone().into());

        macro_rules! after_l_angle {
            () => {{
                let anchor = match self.l_angle_token() {
                    Some(it) => it.into(),
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
                    to_insert.insert(0, make::token(T![,]).into());
                    InsertPosition::After($anchor.syntax().clone().into())
                }
            };
        }

        let position = match self.generic_params().last() {
            Some(it) => after_field!(it),
            None => after_l_angle!(),
        };

        self.insert_children(position, to_insert)
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
        let mut rewriter = SyntaxRewriter::default();
        node.descendants_with_tokens()
            .filter_map(|el| el.into_token())
            .filter_map(ast::Whitespace::cast)
            .filter(|ws| {
                let text = ws.syntax().text();
                text.contains('\n')
            })
            .for_each(|ws| {
                let new_ws = make::tokens::whitespace(&format!("{}{}", ws.syntax(), self,));
                rewriter.replace(ws.syntax(), &new_ws)
            });
        rewriter.rewrite(&node)
    }

    fn decrease_indent(self, node: SyntaxNode) -> SyntaxNode {
        let mut rewriter = SyntaxRewriter::default();
        node.descendants_with_tokens()
            .filter_map(|el| el.into_token())
            .filter_map(ast::Whitespace::cast)
            .filter(|ws| {
                let text = ws.syntax().text();
                text.contains('\n')
            })
            .for_each(|ws| {
                let new_ws = make::tokens::whitespace(
                    &ws.syntax().text().replace(&format!("\n{}", self), "\n"),
                );
                rewriter.replace(ws.syntax(), &new_ws)
            });
        rewriter.rewrite(&node)
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

pub trait AstNodeEdit: AstNode + Clone + Sized {
    #[must_use]
    fn insert_children(
        &self,
        position: InsertPosition<SyntaxElement>,
        to_insert: impl IntoIterator<Item = SyntaxElement>,
    ) -> Self {
        let new_syntax = algo::insert_children(self.syntax(), position, to_insert);
        Self::cast(new_syntax).unwrap()
    }

    #[must_use]
    fn replace_children(
        &self,
        to_replace: RangeInclusive<SyntaxElement>,
        to_insert: impl IntoIterator<Item = SyntaxElement>,
    ) -> Self {
        let new_syntax = algo::replace_children(self.syntax(), to_replace, to_insert);
        Self::cast(new_syntax).unwrap()
    }

    #[must_use]
    fn replace_descendant<D: AstNode>(&self, old: D, new: D) -> Self {
        self.replace_descendants(iter::once((old, new)))
    }

    #[must_use]
    fn replace_descendants<D: AstNode>(
        &self,
        replacement_map: impl IntoIterator<Item = (D, D)>,
    ) -> Self {
        let mut rewriter = SyntaxRewriter::default();
        for (from, to) in replacement_map {
            rewriter.replace(from.syntax(), to.syntax())
        }
        rewriter.rewrite_ast(self)
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
