//! Syntax Tree editor
//!
//! Inspired by Roslyn's [`SyntaxEditor`], but is temporarily built upon mutable syntax tree editing.
//!
//! [`SyntaxEditor`]: https://github.com/dotnet/roslyn/blob/43b0b05cc4f492fd5de00f6f6717409091df8daa/src/Workspaces/Core/Portable/Editing/SyntaxEditor.cs

use std::{
    fmt, iter,
    num::NonZeroU32,
    ops::RangeInclusive,
    sync::atomic::{AtomicU32, Ordering},
};

use rowan::TextRange;
use rustc_hash::FxHashMap;

use crate::{SyntaxElement, SyntaxNode, SyntaxToken};

mod edit_algo;
mod edits;
mod mapping;

pub use edits::Removable;
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
        Self { root, changes: vec![], mappings: SyntaxMapping::default(), annotations: vec![] }
    }

    pub fn add_annotation(&mut self, element: impl Element, annotation: SyntaxAnnotation) {
        self.annotations.push((element.syntax_element(), annotation))
    }

    pub fn add_annotation_all(
        &mut self,
        elements: Vec<impl Element>,
        annotation: SyntaxAnnotation,
    ) {
        self.annotations
            .extend(elements.into_iter().map(|e| e.syntax_element()).zip(iter::repeat(annotation)));
    }

    pub fn merge(&mut self, mut other: SyntaxEditor) {
        debug_assert!(
            self.root == other.root || other.root.ancestors().any(|node| node == self.root),
            "{:?} is not in the same tree as {:?}",
            other.root,
            self.root
        );

        self.changes.append(&mut other.changes);
        self.mappings.merge(other.mappings);
        self.annotations.append(&mut other.annotations);
    }

    pub fn insert(&mut self, position: Position, element: impl Element) {
        debug_assert!(is_ancestor_or_self(&position.parent(), &self.root));
        self.changes.push(Change::Insert(position, element.syntax_element()))
    }

    pub fn insert_all(&mut self, position: Position, elements: Vec<SyntaxElement>) {
        debug_assert!(is_ancestor_or_self(&position.parent(), &self.root));
        self.changes.push(Change::InsertAll(position, elements))
    }

    pub fn delete(&mut self, element: impl Element) {
        let element = element.syntax_element();
        debug_assert!(is_ancestor_or_self_of_element(&element, &self.root));
        debug_assert!(
            !matches!(&element, SyntaxElement::Node(node) if node == &self.root),
            "should not delete root node"
        );
        self.changes.push(Change::Replace(element.syntax_element(), None));
    }

    pub fn delete_all(&mut self, range: RangeInclusive<SyntaxElement>) {
        if range.start() == range.end() {
            self.delete(range.start());
            return;
        }

        debug_assert!(is_ancestor_or_self_of_element(range.start(), &self.root));
        self.changes.push(Change::ReplaceAll(range, Vec::new()))
    }

    pub fn replace(&mut self, old: impl Element, new: impl Element) {
        let old = old.syntax_element();
        debug_assert!(is_ancestor_or_self_of_element(&old, &self.root));
        self.changes.push(Change::Replace(old.syntax_element(), Some(new.syntax_element())));
    }

    pub fn replace_with_many(&mut self, old: impl Element, new: Vec<SyntaxElement>) {
        let old = old.syntax_element();
        debug_assert!(is_ancestor_or_self_of_element(&old, &self.root));
        debug_assert!(
            !(matches!(&old, SyntaxElement::Node(node) if node == &self.root) && new.len() > 1),
            "cannot replace root node with many elements"
        );
        self.changes.push(Change::ReplaceWithMany(old.syntax_element(), new));
    }

    pub fn replace_all(&mut self, range: RangeInclusive<SyntaxElement>, new: Vec<SyntaxElement>) {
        if range.start() == range.end() {
            self.replace_with_many(range.start(), new);
            return;
        }

        debug_assert!(is_ancestor_or_self_of_element(range.start(), &self.root));
        self.changes.push(Change::ReplaceAll(range, new))
    }

    pub fn finish(self) -> SyntaxEdit {
        edit_algo::apply_edits(self)
    }

    pub fn add_mappings(&mut self, other: SyntaxMapping) {
        self.mappings.merge(other);
    }
}

/// Represents a completed [`SyntaxEditor`] operation.
pub struct SyntaxEdit {
    old_root: SyntaxNode,
    new_root: SyntaxNode,
    changed_elements: Vec<SyntaxElement>,
    annotations: FxHashMap<SyntaxAnnotation, Vec<SyntaxElement>>,
}

impl SyntaxEdit {
    /// Root of the initial unmodified syntax tree.
    pub fn old_root(&self) -> &SyntaxNode {
        &self.old_root
    }

    /// Root of the modified syntax tree.
    pub fn new_root(&self) -> &SyntaxNode {
        &self.new_root
    }

    /// Which syntax elements in the modified syntax tree were inserted or
    /// modified as part of the edit.
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

impl Default for SyntaxAnnotation {
    fn default() -> Self {
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

impl Position {
    pub(crate) fn parent(&self) -> SyntaxNode {
        self.place().0
    }

    pub(crate) fn place(&self) -> (SyntaxNode, usize) {
        match &self.repr {
            PositionRepr::FirstChild(parent) => (parent.clone(), 0),
            PositionRepr::After(child) => (child.parent().unwrap(), child.index() + 1),
        }
    }
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
    /// Inserts a single element at the specified position.
    Insert(Position, SyntaxElement),
    /// Inserts many elements in-order at the specified position.
    InsertAll(Position, Vec<SyntaxElement>),
    /// Represents both a replace single element and a delete element operation.
    Replace(SyntaxElement, Option<SyntaxElement>),
    /// Replaces a single element with many elements.
    ReplaceWithMany(SyntaxElement, Vec<SyntaxElement>),
    /// Replaces a range of elements with another list of elements.
    /// Range will always have start != end.
    ReplaceAll(RangeInclusive<SyntaxElement>, Vec<SyntaxElement>),
}

impl Change {
    fn target_range(&self) -> TextRange {
        match self {
            Change::Insert(target, _) | Change::InsertAll(target, _) => match &target.repr {
                PositionRepr::FirstChild(parent) => TextRange::at(
                    parent.first_child_or_token().unwrap().text_range().start(),
                    0.into(),
                ),
                PositionRepr::After(child) => TextRange::at(child.text_range().end(), 0.into()),
            },
            Change::Replace(target, _) | Change::ReplaceWithMany(target, _) => target.text_range(),
            Change::ReplaceAll(range, _) => {
                range.start().text_range().cover(range.end().text_range())
            }
        }
    }

    fn target_parent(&self) -> SyntaxNode {
        match self {
            Change::Insert(target, _) | Change::InsertAll(target, _) => target.parent(),
            Change::Replace(target, _) | Change::ReplaceWithMany(target, _) => match target {
                SyntaxElement::Node(target) => target.parent().unwrap_or_else(|| target.clone()),
                SyntaxElement::Token(target) => target.parent().unwrap(),
            },
            Change::ReplaceAll(target, _) => target.start().parent().unwrap(),
        }
    }

    fn change_kind(&self) -> ChangeKind {
        match self {
            Change::Insert(_, _) | Change::InsertAll(_, _) => ChangeKind::Insert,
            Change::Replace(_, _) | Change::ReplaceWithMany(_, _) => ChangeKind::Replace,
            Change::ReplaceAll(_, _) => ChangeKind::ReplaceRange,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ChangeKind {
    Insert,
    ReplaceRange,
    Replace,
}

impl fmt::Display for Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Change::Insert(position, node_or_token) => {
                let parent = position.parent();
                let mut parent_str = parent.to_string();
                let target_range = self.target_range().start() - parent.text_range().start();

                parent_str.insert_str(
                    target_range.into(),
                    &format!("\x1b[42m{node_or_token}\x1b[0m\x1b[K"),
                );
                f.write_str(&parent_str)
            }
            Change::InsertAll(position, vec) => {
                let parent = position.parent();
                let mut parent_str = parent.to_string();
                let target_range = self.target_range().start() - parent.text_range().start();
                let insertion: String = vec.iter().map(|it| it.to_string()).collect();

                parent_str
                    .insert_str(target_range.into(), &format!("\x1b[42m{insertion}\x1b[0m\x1b[K"));
                f.write_str(&parent_str)
            }
            Change::Replace(old, new) => {
                if let Some(new) = new {
                    write!(f, "\x1b[41m{old}\x1b[42m{new}\x1b[0m\x1b[K")
                } else {
                    write!(f, "\x1b[41m{old}\x1b[0m\x1b[K")
                }
            }
            Change::ReplaceWithMany(old, vec) => {
                let new: String = vec.iter().map(|it| it.to_string()).collect();
                write!(f, "\x1b[41m{old}\x1b[42m{new}\x1b[0m\x1b[K")
            }
            Change::ReplaceAll(range, vec) => {
                let parent = range.start().parent().unwrap();
                let parent_str = parent.to_string();
                let pre_range =
                    TextRange::new(parent.text_range().start(), range.start().text_range().start());
                let old_range = TextRange::new(
                    range.start().text_range().start(),
                    range.end().text_range().end(),
                );
                let post_range =
                    TextRange::new(range.end().text_range().end(), parent.text_range().end());

                let pre_str = &parent_str[pre_range - parent.text_range().start()];
                let old_str = &parent_str[old_range - parent.text_range().start()];
                let post_str = &parent_str[post_range - parent.text_range().start()];
                let new: String = vec.iter().map(|it| it.to_string()).collect();

                write!(f, "{pre_str}\x1b[41m{old_str}\x1b[42m{new}\x1b[0m\x1b[K{post_str}")
            }
        }
    }
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

fn is_ancestor_or_self(node: &SyntaxNode, ancestor: &SyntaxNode) -> bool {
    node == ancestor || node.ancestors().any(|it| &it == ancestor)
}

fn is_ancestor_or_self_of_element(node: &SyntaxElement, ancestor: &SyntaxNode) -> bool {
    matches!(node, SyntaxElement::Node(node) if node == ancestor)
        || node.ancestors().any(|it| &it == ancestor)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::{
        AstNode,
        ast::{self, make, syntax_factory::SyntaxFactory},
    };

    use super::*;

    #[test]
    fn basic_usage() {
        let root = make::match_arm(
            make::wildcard_pat().into(),
            None,
            make::expr_tuple([
                make::expr_bin_op(
                    make::expr_literal("2").into(),
                    ast::BinaryOp::ArithOp(ast::ArithOp::Add),
                    make::expr_literal("2").into(),
                ),
                make::expr_literal("true").into(),
            ])
            .into(),
        );

        let to_wrap = root.syntax().descendants().find_map(ast::TupleExpr::cast).unwrap();
        let to_replace = root.syntax().descendants().find_map(ast::BinExpr::cast).unwrap();

        let mut editor = SyntaxEditor::new(root.syntax().clone());
        let make = SyntaxFactory::with_mappings();

        let name = make::name("var_name");
        let name_ref = make::name_ref("var_name").clone_for_update();

        let placeholder_snippet = SyntaxAnnotation::default();
        editor.add_annotation(name.syntax(), placeholder_snippet);
        editor.add_annotation(name_ref.syntax(), placeholder_snippet);

        let new_block = make.block_expr(
            [make
                .let_stmt(
                    make.ident_pat(false, false, name.clone()).into(),
                    None,
                    Some(to_replace.clone().into()),
                )
                .into()],
            Some(to_wrap.clone().into()),
        );

        editor.replace(to_replace.syntax(), name_ref.syntax());
        editor.replace(to_wrap.syntax(), new_block.syntax());
        editor.add_mappings(make.finish_with_mappings());

        let edit = editor.finish();

        let expect = expect![[r#"
            _ => {
                let var_name = 2 + 2;
                (var_name, true)
            },"#]];
        expect.assert_eq(&edit.new_root.to_string());

        assert_eq!(edit.find_annotation(placeholder_snippet).len(), 2);
        assert!(
            edit.annotations
                .iter()
                .flat_map(|(_, elements)| elements)
                .all(|element| element.ancestors().any(|it| &it == edit.new_root()))
        )
    }

    #[test]
    fn test_insert_independent() {
        let root = make::block_expr(
            [make::let_stmt(
                make::ext::simple_ident_pat(make::name("second")).into(),
                None,
                Some(make::expr_literal("2").into()),
            )
            .into()],
            None,
        );

        let second_let = root.syntax().descendants().find_map(ast::LetStmt::cast).unwrap();

        let mut editor = SyntaxEditor::new(root.syntax().clone());
        let make = SyntaxFactory::without_mappings();

        editor.insert(
            Position::first_child_of(root.stmt_list().unwrap().syntax()),
            make.let_stmt(
                make::ext::simple_ident_pat(make::name("first")).into(),
                None,
                Some(make::expr_literal("1").into()),
            )
            .syntax(),
        );

        editor.insert(
            Position::after(second_let.syntax()),
            make.let_stmt(
                make::ext::simple_ident_pat(make::name("third")).into(),
                None,
                Some(make::expr_literal("3").into()),
            )
            .syntax(),
        );

        let edit = editor.finish();

        let expect = expect![[r#"
            let first = 1;{
                let second = 2;let third = 3;
            }"#]];
        expect.assert_eq(&edit.new_root.to_string());
    }

    #[test]
    fn test_insert_dependent() {
        let root = make::block_expr(
            [],
            Some(
                make::block_expr(
                    [make::let_stmt(
                        make::ext::simple_ident_pat(make::name("second")).into(),
                        None,
                        Some(make::expr_literal("2").into()),
                    )
                    .into()],
                    None,
                )
                .into(),
            ),
        );

        let inner_block =
            root.syntax().descendants().flat_map(ast::BlockExpr::cast).nth(1).unwrap();
        let second_let = root.syntax().descendants().find_map(ast::LetStmt::cast).unwrap();

        let mut editor = SyntaxEditor::new(root.syntax().clone());
        let make = SyntaxFactory::with_mappings();

        let new_block_expr = make.block_expr([], Some(ast::Expr::BlockExpr(inner_block.clone())));

        let first_let = make.let_stmt(
            make::ext::simple_ident_pat(make::name("first")).into(),
            None,
            Some(make::expr_literal("1").into()),
        );

        let third_let = make.let_stmt(
            make::ext::simple_ident_pat(make::name("third")).into(),
            None,
            Some(make::expr_literal("3").into()),
        );

        editor.insert(
            Position::first_child_of(inner_block.stmt_list().unwrap().syntax()),
            first_let.syntax(),
        );
        editor.insert(Position::after(second_let.syntax()), third_let.syntax());
        editor.replace(inner_block.syntax(), new_block_expr.syntax());
        editor.add_mappings(make.finish_with_mappings());

        let edit = editor.finish();

        let expect = expect![[r#"
            {
                {
                let first = 1;{
                let second = 2;let third = 3;
            }
            }
            }"#]];
        expect.assert_eq(&edit.new_root.to_string());
    }

    #[test]
    fn test_replace_root_with_dependent() {
        let root = make::block_expr(
            [make::let_stmt(
                make::ext::simple_ident_pat(make::name("second")).into(),
                None,
                Some(make::expr_literal("2").into()),
            )
            .into()],
            None,
        );

        let inner_block = root.clone();

        let mut editor = SyntaxEditor::new(root.syntax().clone());
        let make = SyntaxFactory::with_mappings();

        let new_block_expr = make.block_expr([], Some(ast::Expr::BlockExpr(inner_block.clone())));

        let first_let = make.let_stmt(
            make::ext::simple_ident_pat(make::name("first")).into(),
            None,
            Some(make::expr_literal("1").into()),
        );

        editor.insert(
            Position::first_child_of(inner_block.stmt_list().unwrap().syntax()),
            first_let.syntax(),
        );
        editor.replace(inner_block.syntax(), new_block_expr.syntax());
        editor.add_mappings(make.finish_with_mappings());

        let edit = editor.finish();

        let expect = expect![[r#"
            {
                let first = 1;{
                let second = 2;
            }
            }"#]];
        expect.assert_eq(&edit.new_root.to_string());
    }

    #[test]
    fn test_replace_token_in_parent() {
        let parent_fn = make::fn_(
            None,
            make::name("it"),
            None,
            None,
            make::param_list(None, []),
            make::block_expr([], Some(make::ext::expr_unit())),
            Some(make::ret_type(make::ty_unit())),
            false,
            false,
            false,
            false,
        );

        let mut editor = SyntaxEditor::new(parent_fn.syntax().clone());

        if let Some(ret_ty) = parent_fn.ret_type() {
            editor.delete(ret_ty.syntax().clone());

            if let Some(SyntaxElement::Token(token)) = ret_ty.syntax().next_sibling_or_token()
                && token.kind().is_trivia()
            {
                editor.delete(token);
            }
        }

        if let Some(tail) = parent_fn.body().unwrap().tail_expr() {
            editor.delete(tail.syntax().clone());
        }

        let edit = editor.finish();

        let expect = expect![["fn it() {\n    \n}"]];
        expect.assert_eq(&edit.new_root.to_string());
    }
}
