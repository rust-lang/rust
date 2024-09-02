//! Syntax Tree editor
//!
//! Inspired by Roslyn's [`SyntaxEditor`], but is temporarily built upon mutable syntax tree editing.
//!
//! [`SyntaxEditor`]: https://github.com/dotnet/roslyn/blob/43b0b05cc4f492fd5de00f6f6717409091df8daa/src/Workspaces/Core/Portable/Editing/SyntaxEditor.cs

use std::{
    num::NonZeroU32,
    sync::atomic::{AtomicU32, Ordering},
};

use rowan::TextRange;
use rustc_hash::FxHashMap;

use crate::{SyntaxElement, SyntaxNode, SyntaxToken};

mod edit_algo;
mod mapping;

pub use mapping::{SyntaxMapping, SyntaxMappingBuilder};

#[derive(Debug)]
pub struct SyntaxEditor {
    root: SyntaxNode,
    changes: Vec<Change>,
    mappings: SyntaxMapping,
    annotations: Vec<(SyntaxElement, SyntaxAnnotation)>,
}

impl SyntaxEditor {
    /// Creates a syntax editor to start editing from `root`
    pub fn new(root: SyntaxNode) -> Self {
        Self { root, changes: vec![], mappings: SyntaxMapping::new(), annotations: vec![] }
    }

    pub fn add_annotation(&mut self, element: impl Element, annotation: SyntaxAnnotation) {
        self.annotations.push((element.syntax_element(), annotation))
    }

    pub fn combine(&mut self, other: SyntaxEditor) {
        todo!()
    }

    pub fn delete(&mut self, element: impl Element) {
        self.changes.push(Change::Replace(element.syntax_element(), None));
    }

    pub fn replace(&mut self, old: impl Element, new: impl Element) {
        self.changes.push(Change::Replace(old.syntax_element(), Some(new.syntax_element())));
    }

    pub fn finish(self) -> SyntaxEdit {
        edit_algo::apply_edits(self)
    }
}

pub struct SyntaxEdit {
    root: SyntaxNode,
    changed_elements: Vec<SyntaxElement>,
    annotations: FxHashMap<SyntaxAnnotation, Vec<SyntaxElement>>,
}

impl SyntaxEdit {
    /// Root of the modified syntax tree
    pub fn root(&self) -> &SyntaxNode {
        &self.root
    }

    /// Which syntax elements in the modified syntax tree were modified as part
    /// of the edit.
    ///
    /// Note that for syntax nodes, only the upper-most parent of a set of
    /// changes is included, not any child elements that may have been modified.
    pub fn changed_elements(&self) -> &[SyntaxElement] {
        self.changed_elements.as_slice()
    }

    /// Finds which syntax elements have been annotated with the given
    /// annotation.
    ///
    /// Note that an annotation might not appear in the modified syntax tree if
    /// the syntax elements that were annotated did not make it into the final
    /// syntax tree.
    pub fn find_annotation(&self, annotation: SyntaxAnnotation) -> &[SyntaxElement] {
        self.annotations.get(&annotation).as_ref().map_or(&[], |it| it.as_slice())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SyntaxAnnotation(NonZeroU32);

impl SyntaxAnnotation {
    /// Creates a unique syntax annotation to attach data to.
    pub fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(1);

        // Only consistency within a thread matters, as SyntaxElements are !Send
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);

        Self(NonZeroU32::new(id).expect("syntax annotation id overflow"))
    }
}

/// Position describing where to insert elements
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
}

#[derive(Debug)]
enum Change {
    /// Represents both a replace single element and a delete element operation.
    Replace(SyntaxElement, Option<SyntaxElement>),
}

impl Change {
    fn target_range(&self) -> TextRange {
        match self {
            Change::Replace(target, _) => target.text_range(),
        }
    }

    fn target_parent(&self) -> SyntaxNode {
        match self {
            Change::Replace(target, _) => target.parent().unwrap(),
        }
    }

    fn change_kind(&self) -> ChangeKind {
        match self {
            Change::Replace(_, _) => ChangeKind::Replace,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ChangeKind {
    Insert,
    // TODO: deal with replace spans
    Replace,
}

/// Utility trait to allow calling syntax editor functions with references or owned
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

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use itertools::Itertools;

    use crate::{
        ast::{self, make, HasName},
        AstNode,
    };

    use super::*;

    fn make_ident_pat(
        editor: Option<&mut SyntaxEditor>,
        ref_: bool,
        mut_: bool,
        name: ast::Name,
    ) -> ast::IdentPat {
        let ast = make::ident_pat(ref_, mut_, name.clone()).clone_for_update();

        if let Some(editor) = editor {
            let mut mapping = SyntaxMappingBuilder::new(ast.syntax().clone());
            mapping.map_node(name.syntax().clone(), ast.name().unwrap().syntax().clone());
            mapping.finish(editor);
        }

        ast
    }

    fn make_let_stmt(
        editor: Option<&mut SyntaxEditor>,
        pattern: ast::Pat,
        ty: Option<ast::Type>,
        initializer: Option<ast::Expr>,
    ) -> ast::LetStmt {
        let ast =
            make::let_stmt(pattern.clone(), ty.clone(), initializer.clone()).clone_for_update();

        if let Some(editor) = editor {
            let mut mapping = SyntaxMappingBuilder::new(ast.syntax().clone());
            mapping.map_node(pattern.syntax().clone(), ast.pat().unwrap().syntax().clone());
            if let Some(input) = ty {
                mapping.map_node(input.syntax().clone(), ast.ty().unwrap().syntax().clone());
            }
            if let Some(input) = initializer {
                mapping
                    .map_node(input.syntax().clone(), ast.initializer().unwrap().syntax().clone());
            }
            mapping.finish(editor);
        }

        ast
    }

    fn make_block_expr(
        editor: Option<&mut SyntaxEditor>,
        stmts: impl IntoIterator<Item = ast::Stmt>,
        tail_expr: Option<ast::Expr>,
    ) -> ast::BlockExpr {
        let stmts = stmts.into_iter().collect_vec();
        let input = stmts.iter().map(|it| it.syntax().clone()).collect_vec();

        let ast = make::block_expr(stmts, tail_expr.clone()).clone_for_update();

        if let Some((editor, stmt_list)) = editor.zip(ast.stmt_list()) {
            let mut mapping = SyntaxMappingBuilder::new(stmt_list.syntax().clone());

            mapping.map_children(
                input.into_iter(),
                stmt_list.statements().map(|it| it.syntax().clone()),
            );

            if let Some((input, output)) = tail_expr.zip(stmt_list.tail_expr()) {
                mapping.map_node(input.syntax().clone(), output.syntax().clone());
            }

            mapping.finish(editor);
        }

        ast
    }

    #[test]
    fn it() {
        let root = make::match_arm(
            [make::wildcard_pat().into()],
            None,
            make::expr_tuple([
                make::expr_bin_op(
                    make::expr_literal("2").into(),
                    ast::BinaryOp::ArithOp(ast::ArithOp::Add),
                    make::expr_literal("2").into(),
                ),
                make::expr_literal("true").into(),
            ]),
        );

        let to_wrap = root.syntax().descendants().find_map(ast::TupleExpr::cast).unwrap();
        let to_replace = root.syntax().descendants().find_map(ast::BinExpr::cast).unwrap();

        let mut editor = SyntaxEditor::new(root.syntax().clone());

        let name = make::name("var_name");
        let name_ref = make::name_ref("var_name").clone_for_update();

        let placeholder_snippet = SyntaxAnnotation::new();
        editor.add_annotation(name.syntax(), placeholder_snippet);
        editor.add_annotation(name_ref.syntax(), placeholder_snippet);

        let make_ident_pat = make_ident_pat(Some(&mut editor), false, false, name);
        let make_let_stmt = make_let_stmt(
            Some(&mut editor),
            make_ident_pat.into(),
            None,
            Some(to_replace.clone().into()),
        );
        let new_block = make_block_expr(
            Some(&mut editor),
            [make_let_stmt.into()],
            Some(to_wrap.clone().into()),
        );

        // should die:
        editor.replace(to_replace.syntax(), name_ref.syntax());
        editor.replace(to_wrap.syntax(), new_block.syntax());
        // editor.replace(to_replace.syntax(), name_ref.syntax());

        // dbg!(&editor.mappings);
        let edit = editor.finish();

        let expect = expect![];
        expect.assert_eq(&edit.root.to_string());
        assert_eq!(edit.find_annotation(placeholder_snippet).len(), 2);
    }
}
