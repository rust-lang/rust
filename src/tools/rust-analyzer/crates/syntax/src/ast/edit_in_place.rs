//! Structural editing for ast.

use std::iter::{empty, once, successors};

use parser::T;

use crate::{
    AstNode, AstToken, Direction,
    algo::{self, neighbor},
    ast::{self, make, syntax_factory::SyntaxFactory},
    syntax_editor::SyntaxEditor,
    ted,
};

use super::HasName;

impl ast::GenericParamList {
    /// Constructs a matching [`ast::GenericArgList`]
    pub fn to_generic_args(&self, make: &SyntaxFactory) -> ast::GenericArgList {
        let args = self.generic_params().filter_map(|param| match param {
            ast::GenericParam::LifetimeParam(it) => {
                Some(ast::GenericArg::LifetimeArg(make.lifetime_arg(it.lifetime()?)))
            }
            ast::GenericParam::TypeParam(it) => {
                Some(ast::GenericArg::TypeArg(make.type_arg(make.ty_name(it.name()?))))
            }
            ast::GenericParam::ConstParam(it) => {
                // Name-only const params get parsed as `TypeArg`s
                Some(ast::GenericArg::TypeArg(make.type_arg(make.ty_name(it.name()?))))
            }
        });

        make::generic_arg_list(args)
    }
}

pub trait Removable: AstNode {
    fn remove(&self);
}

impl Removable for ast::UseTree {
    fn remove(&self) {
        for dir in [Direction::Next, Direction::Prev] {
            if let Some(next_use_tree) = neighbor(self, dir) {
                let separators = self
                    .syntax()
                    .siblings_with_tokens(dir)
                    .skip(1)
                    .take_while(|it| it.as_node() != Some(next_use_tree.syntax()));
                ted::remove_all_iter(separators);
                break;
            }
        }
        ted::remove(self.syntax());
    }
}

impl ast::UseTree {
    /// Editor variant of UseTree remove
    fn remove_with_editor(&self, editor: &SyntaxEditor) {
        for dir in [Direction::Next, Direction::Prev] {
            if let Some(next_use_tree) = neighbor(self, dir) {
                let separators = self
                    .syntax()
                    .siblings_with_tokens(dir)
                    .skip(1)
                    .take_while(|it| it.as_node() != Some(next_use_tree.syntax()));
                for separator in separators {
                    editor.delete(separator);
                }
                break;
            }
        }
        editor.delete(self.syntax());
    }

    /// Deletes the usetree node represented by the input. Recursively removes parents, including use nodes that become empty.
    pub fn remove_recursive(self, editor: &SyntaxEditor) {
        let parent = self.syntax().parent();

        if let Some(u) = parent.clone().and_then(ast::Use::cast) {
            u.remove(editor);
        } else if let Some(u) = parent.and_then(ast::UseTreeList::cast) {
            if u.use_trees().nth(1).is_none()
                || u.use_trees().all(|use_tree| {
                    use_tree.syntax() == self.syntax() || editor.deleted(use_tree.syntax())
                })
            {
                u.parent_use_tree().remove_recursive(editor);
                return;
            }
            self.remove_with_editor(editor);
            u.remove_unnecessary_braces(editor);
        }
    }

    pub fn get_or_create_use_tree_list(&self) -> ast::UseTreeList {
        match self.use_tree_list() {
            Some(it) => it,
            None => {
                let position = ted::Position::last_child_of(self.syntax());
                let use_tree_list = make::use_tree_list(empty()).clone_for_update();
                let mut elements = Vec::with_capacity(2);
                if self.coloncolon_token().is_none() {
                    elements.push(make::token(T![::]).into());
                }
                elements.push(use_tree_list.syntax().clone().into());
                ted::insert_all_raw(position, elements);
                use_tree_list
            }
        }
    }

    /// Splits off the given prefix, making it the path component of the use tree,
    /// appending the rest of the path to all UseTreeList items.
    ///
    /// # Examples
    ///
    /// `prefix$0::suffix` -> `prefix::{suffix}`
    ///
    /// `prefix$0` -> `prefix::{self}`
    ///
    /// `prefix$0::*` -> `prefix::{*}`
    pub fn split_prefix(&self, prefix: &ast::Path) {
        debug_assert_eq!(self.path(), Some(prefix.top_path()));
        let path = self.path().unwrap();
        if &path == prefix && self.use_tree_list().is_none() {
            if self.star_token().is_some() {
                // path$0::* -> *
                if let Some(a) = self.coloncolon_token() {
                    ted::remove(a)
                }
                ted::remove(prefix.syntax());
            } else {
                // path$0 -> self
                let self_suffix =
                    make::path_unqualified(make::path_segment_self()).clone_for_update();
                ted::replace(path.syntax(), self_suffix.syntax());
            }
        } else if split_path_prefix(prefix).is_none() {
            return;
        }
        // At this point, prefix path is detached; _self_ use tree has suffix path.
        // Next, transform 'suffix' use tree into 'prefix::{suffix}'
        let subtree = self.clone_subtree().clone_for_update();
        ted::remove_all_iter(self.syntax().children_with_tokens());
        ted::insert(ted::Position::first_child_of(self.syntax()), prefix.syntax());
        self.get_or_create_use_tree_list().add_use_tree(subtree);

        fn split_path_prefix(prefix: &ast::Path) -> Option<()> {
            let parent = prefix.parent_path()?;
            let segment = parent.segment()?;
            if algo::has_errors(segment.syntax()) {
                return None;
            }
            for p in successors(parent.parent_path(), |it| it.parent_path()) {
                p.segment()?;
            }
            if let Some(a) = prefix.parent_path().and_then(|p| p.coloncolon_token()) {
                ted::remove(a)
            }
            ted::remove(prefix.syntax());
            Some(())
        }
    }

    /// Editor variant of `split_prefix`
    pub fn split_prefix_with_editor(&self, editor: &SyntaxEditor, prefix: &ast::Path) {
        debug_assert_eq!(self.path(), Some(prefix.top_path()));

        let make = editor.make();
        let path = self.path().unwrap();
        let suffix = if path == *prefix && self.use_tree_list().is_none() {
            if self.star_token().is_some() {
                make.use_tree_glob()
            } else {
                let self_path = make.path_unqualified(make.path_segment_self());
                make.use_tree(self_path, None, None, false)
            }
        } else {
            let suffix_segments = path.segments().skip(prefix.segments().count());
            let suffix_path = make.path_from_segments(suffix_segments, false);
            make.use_tree(
                suffix_path,
                self.use_tree_list(),
                self.rename(),
                self.star_token().is_some(),
            )
        };
        let use_tree_list = make.use_tree_list(once(suffix));
        let new_use_tree = make.use_tree(prefix.clone(), Some(use_tree_list), None, false);

        editor.replace(self.syntax(), new_use_tree.syntax());
    }

    /// Wraps the use tree in use tree list with no top level path (if it isn't already).
    ///
    /// # Examples
    ///
    /// `foo::bar` -> `{foo::bar}`
    ///
    /// `{foo::bar}` -> `{foo::bar}`
    pub fn wrap_in_tree_list(&self) -> Option<()> {
        if self.use_tree_list().is_some()
            && self.path().is_none()
            && self.star_token().is_none()
            && self.rename().is_none()
        {
            return None;
        }
        let subtree = self.clone_subtree().clone_for_update();
        ted::remove_all_iter(self.syntax().children_with_tokens());
        ted::append_child(
            self.syntax(),
            make::use_tree_list(once(subtree)).clone_for_update().syntax(),
        );
        Some(())
    }
}

impl ast::UseTreeList {
    pub fn add_use_tree(&self, use_tree: ast::UseTree) {
        let (position, elements) = match self.use_trees().last() {
            Some(last_tree) => (
                ted::Position::after(last_tree.syntax()),
                vec![
                    make::token(T![,]).into(),
                    make::tokens::single_space().into(),
                    use_tree.syntax.into(),
                ],
            ),
            None => {
                let position = match self.l_curly_token() {
                    Some(l_curly) => ted::Position::after(l_curly),
                    None => ted::Position::last_child_of(self.syntax()),
                };
                (position, vec![use_tree.syntax.into()])
            }
        };
        ted::insert_all_raw(position, elements);
    }
}

impl ast::Use {
    fn remove(&self, editor: &SyntaxEditor) {
        let make = editor.make();
        let next_ws = self
            .syntax()
            .next_sibling_or_token()
            .and_then(|it| it.into_token())
            .and_then(ast::Whitespace::cast);
        if let Some(next_ws) = next_ws {
            let ws_text = next_ws.syntax().text();
            if let Some(rest) = ws_text.strip_prefix('\n') {
                let next_use_removed = next_ws
                    .syntax()
                    .next_sibling_or_token()
                    .and_then(|it| it.into_node())
                    .and_then(ast::Use::cast)
                    .and_then(|use_| use_.use_tree())
                    .is_some_and(|use_tree| editor.deleted(use_tree.syntax()));
                if rest.is_empty() || next_use_removed {
                    editor.delete(next_ws.syntax());
                } else {
                    editor.replace(next_ws.syntax(), make.whitespace(rest));
                }
            }
        }
        let prev_ws = self
            .syntax()
            .prev_sibling_or_token()
            .and_then(|it| it.into_token())
            .and_then(ast::Whitespace::cast);
        if let Some(prev_ws) = prev_ws {
            let ws_text = prev_ws.syntax().text();
            let prev_newline = ws_text.rfind('\n').map(|x| x + 1).unwrap_or(0);
            let rest = &ws_text[0..prev_newline];
            if rest.is_empty() {
                editor.delete(prev_ws.syntax());
            } else {
                editor.replace(prev_ws.syntax(), make.whitespace(rest));
            }
        }

        editor.delete(self.syntax());
    }
}

impl ast::Impl {
    pub fn get_or_create_assoc_item_list(&self) -> ast::AssocItemList {
        if self.assoc_item_list().is_none() {
            let assoc_item_list = make::assoc_item_list(None).clone_for_update();
            ted::append_child(self.syntax(), assoc_item_list.syntax());
        }
        self.assoc_item_list().unwrap()
    }
}

impl ast::RecordExprField {
    /// This will either replace the initializer, or in the case that this is a shorthand convert
    /// the initializer into the name ref and insert the expr as the new initializer.
    pub fn replace_expr(&self, editor: &SyntaxEditor, expr: ast::Expr) {
        if self.name_ref().is_some() {
            if let Some(prev) = self.expr() {
                editor.replace(prev.syntax(), expr.syntax());
            }
        } else if let Some(ast::Expr::PathExpr(path_expr)) = self.expr()
            && let Some(path) = path_expr.path()
            && let Some(name_ref) = path.as_single_name_ref()
        {
            // shorthand `{ x }` → expand to `{ x: expr }`
            let new_field = editor
                .make()
                .record_expr_field(editor.make().name_ref(&name_ref.text()), Some(expr));
            editor.replace(self.syntax(), new_field.syntax());
        }
    }
}
