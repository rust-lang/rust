//! Structural editing for ast.

use std::iter::{empty, once, successors};

use parser::{SyntaxKind, T};

use crate::{
    AstNode, AstToken, Direction, SyntaxElement,
    SyntaxKind::{ATTR, COMMENT, WHITESPACE},
    SyntaxNode, SyntaxToken,
    algo::{self, neighbor},
    ast::{self, HasGenericParams, edit::IndentLevel, make},
    ted::{self, Position},
};

use super::{GenericParam, HasName};

pub trait GenericParamsOwnerEdit: ast::HasGenericParams {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList;
    fn get_or_create_where_clause(&self) -> ast::WhereClause;
}

impl GenericParamsOwnerEdit for ast::Fn {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Position::after(name.syntax)
                } else if let Some(fn_token) = self.fn_token() {
                    Position::after(fn_token)
                } else if let Some(param_list) = self.param_list() {
                    Position::before(param_list.syntax)
                } else {
                    Position::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            }
        }
    }

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(ty) = self.ret_type() {
                Position::after(ty.syntax())
            } else if let Some(param_list) = self.param_list() {
                Position::after(param_list.syntax())
            } else {
                Position::last_child_of(self.syntax())
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Impl {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = match self.impl_token() {
                    Some(imp_token) => Position::after(imp_token),
                    None => Position::last_child_of(self.syntax()),
                };
                create_generic_param_list(position)
            }
        }
    }

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = match self.assoc_item_list() {
                Some(items) => Position::before(items.syntax()),
                None => Position::last_child_of(self.syntax()),
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Trait {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Position::after(name.syntax)
                } else if let Some(trait_token) = self.trait_token() {
                    Position::after(trait_token)
                } else {
                    Position::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            }
        }
    }

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = match self.assoc_item_list() {
                Some(items) => Position::before(items.syntax()),
                None => Position::last_child_of(self.syntax()),
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::TraitAlias {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Position::after(name.syntax)
                } else if let Some(trait_token) = self.trait_token() {
                    Position::after(trait_token)
                } else {
                    Position::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            }
        }
    }

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = match self.semicolon_token() {
                Some(tok) => Position::before(tok),
                None => Position::last_child_of(self.syntax()),
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::TypeAlias {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Position::after(name.syntax)
                } else if let Some(trait_token) = self.type_token() {
                    Position::after(trait_token)
                } else {
                    Position::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            }
        }
    }

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = match self.eq_token() {
                Some(tok) => Position::before(tok),
                None => match self.semicolon_token() {
                    Some(tok) => Position::before(tok),
                    None => Position::last_child_of(self.syntax()),
                },
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Struct {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Position::after(name.syntax)
                } else if let Some(struct_token) = self.struct_token() {
                    Position::after(struct_token)
                } else {
                    Position::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            }
        }
    }

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let tfl = self.field_list().and_then(|fl| match fl {
                ast::FieldList::RecordFieldList(_) => None,
                ast::FieldList::TupleFieldList(it) => Some(it),
            });
            let position = if let Some(tfl) = tfl {
                Position::after(tfl.syntax())
            } else if let Some(gpl) = self.generic_param_list() {
                Position::after(gpl.syntax())
            } else if let Some(name) = self.name() {
                Position::after(name.syntax())
            } else {
                Position::last_child_of(self.syntax())
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Enum {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Position::after(name.syntax)
                } else if let Some(enum_token) = self.enum_token() {
                    Position::after(enum_token)
                } else {
                    Position::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            }
        }
    }

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(gpl) = self.generic_param_list() {
                Position::after(gpl.syntax())
            } else if let Some(name) = self.name() {
                Position::after(name.syntax())
            } else {
                Position::last_child_of(self.syntax())
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}

fn create_where_clause(position: Position) {
    let where_clause = make::where_clause(empty()).clone_for_update();
    ted::insert(position, where_clause.syntax());
}

fn create_generic_param_list(position: Position) -> ast::GenericParamList {
    let gpl = make::generic_param_list(empty()).clone_for_update();
    ted::insert_raw(position, gpl.syntax());
    gpl
}

pub trait AttrsOwnerEdit: ast::HasAttrs {
    fn remove_attrs_and_docs(&self) {
        remove_attrs_and_docs(self.syntax());

        fn remove_attrs_and_docs(node: &SyntaxNode) {
            let mut remove_next_ws = false;
            for child in node.children_with_tokens() {
                match child.kind() {
                    ATTR | COMMENT => {
                        remove_next_ws = true;
                        child.detach();
                        continue;
                    }
                    WHITESPACE if remove_next_ws => {
                        child.detach();
                    }
                    _ => (),
                }
                remove_next_ws = false;
            }
        }
    }

    fn add_attr(&self, attr: ast::Attr) {
        add_attr(self.syntax(), attr);

        fn add_attr(node: &SyntaxNode, attr: ast::Attr) {
            let indent = IndentLevel::from_node(node);
            attr.reindent_to(indent);

            let after_attrs_and_comments = node
                .children_with_tokens()
                .find(|it| !matches!(it.kind(), WHITESPACE | COMMENT | ATTR))
                .map_or(Position::first_child_of(node), Position::before);

            ted::insert_all(
                after_attrs_and_comments,
                vec![
                    attr.syntax().clone().into(),
                    make::tokens::whitespace(&format!("\n{indent}")).into(),
                ],
            )
        }
    }
}

impl<T: ast::HasAttrs> AttrsOwnerEdit for T {}

impl ast::GenericParamList {
    pub fn add_generic_param(&self, generic_param: ast::GenericParam) {
        match self.generic_params().last() {
            Some(last_param) => {
                let position = Position::after(last_param.syntax());
                let elements = vec![
                    make::token(T![,]).into(),
                    make::tokens::single_space().into(),
                    generic_param.syntax().clone().into(),
                ];
                ted::insert_all(position, elements);
            }
            None => {
                let after_l_angle = Position::after(self.l_angle_token().unwrap());
                ted::insert(after_l_angle, generic_param.syntax());
            }
        }
    }

    /// Removes the existing generic param
    pub fn remove_generic_param(&self, generic_param: ast::GenericParam) {
        if let Some(previous) = generic_param.syntax().prev_sibling() {
            if let Some(next_token) = previous.next_sibling_or_token() {
                ted::remove_all(next_token..=generic_param.syntax().clone().into());
            }
        } else if let Some(next) = generic_param.syntax().next_sibling() {
            if let Some(next_token) = next.prev_sibling_or_token() {
                ted::remove_all(generic_param.syntax().clone().into()..=next_token);
            }
        } else {
            ted::remove(generic_param.syntax());
        }
    }

    /// Find the params corresponded to generic arg
    pub fn find_generic_arg(&self, generic_arg: &ast::GenericArg) -> Option<GenericParam> {
        self.generic_params().find_map(move |param| match (&param, &generic_arg) {
            (ast::GenericParam::LifetimeParam(a), ast::GenericArg::LifetimeArg(b)) => {
                (a.lifetime()?.lifetime_ident_token()?.text()
                    == b.lifetime()?.lifetime_ident_token()?.text())
                .then_some(param)
            }
            (ast::GenericParam::TypeParam(a), ast::GenericArg::TypeArg(b)) => {
                debug_assert_eq!(b.syntax().first_token(), b.syntax().last_token());
                (a.name()?.text() == b.syntax().first_token()?.text()).then_some(param)
            }
            (ast::GenericParam::ConstParam(a), ast::GenericArg::TypeArg(b)) => {
                debug_assert_eq!(b.syntax().first_token(), b.syntax().last_token());
                (a.name()?.text() == b.syntax().first_token()?.text()).then_some(param)
            }
            _ => None,
        })
    }

    /// Removes the corresponding generic arg
    pub fn remove_generic_arg(&self, generic_arg: &ast::GenericArg) {
        let param_to_remove = self.find_generic_arg(generic_arg);

        if let Some(param) = &param_to_remove {
            self.remove_generic_param(param.clone());
        }
    }

    /// Constructs a matching [`ast::GenericArgList`]
    pub fn to_generic_args(&self) -> ast::GenericArgList {
        let args = self.generic_params().filter_map(|param| match param {
            ast::GenericParam::LifetimeParam(it) => {
                Some(ast::GenericArg::LifetimeArg(make::lifetime_arg(it.lifetime()?)))
            }
            ast::GenericParam::TypeParam(it) => {
                Some(ast::GenericArg::TypeArg(make::type_arg(make::ext::ty_name(it.name()?))))
            }
            ast::GenericParam::ConstParam(it) => {
                // Name-only const params get parsed as `TypeArg`s
                Some(ast::GenericArg::TypeArg(make::type_arg(make::ext::ty_name(it.name()?))))
            }
        });

        make::generic_arg_list(args)
    }
}

impl ast::WhereClause {
    pub fn add_predicate(&self, predicate: ast::WherePred) {
        if let Some(pred) = self.predicates().last()
            && !pred.syntax().siblings_with_tokens(Direction::Next).any(|it| it.kind() == T![,])
        {
            ted::append_child_raw(self.syntax(), make::token(T![,]));
        }
        ted::append_child(self.syntax(), predicate.syntax());
    }

    pub fn remove_predicate(&self, predicate: ast::WherePred) {
        if let Some(previous) = predicate.syntax().prev_sibling() {
            if let Some(next_token) = previous.next_sibling_or_token() {
                ted::remove_all(next_token..=predicate.syntax().clone().into());
            }
        } else if let Some(next) = predicate.syntax().next_sibling() {
            if let Some(next_token) = next.prev_sibling_or_token() {
                ted::remove_all(predicate.syntax().clone().into()..=next_token);
            }
        } else {
            ted::remove(predicate.syntax());
        }
    }
}

pub trait Removable: AstNode {
    fn remove(&self);
}

impl Removable for ast::TypeBoundList {
    fn remove(&self) {
        match self.syntax().siblings_with_tokens(Direction::Prev).find(|it| it.kind() == T![:]) {
            Some(colon) => ted::remove_all(colon..=self.syntax().clone().into()),
            None => ted::remove(self.syntax()),
        }
    }
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
    /// Deletes the usetree node represented by the input. Recursively removes parents, including use nodes that become empty.
    pub fn remove_recursive(self) {
        let parent = self.syntax().parent();

        self.remove();

        if let Some(u) = parent.clone().and_then(ast::Use::cast) {
            if u.use_tree().is_none() {
                u.remove();
            }
        } else if let Some(u) = parent.and_then(ast::UseTreeList::cast) {
            if u.use_trees().next().is_none() {
                let parent = u.syntax().parent().and_then(ast::UseTree::cast);
                if let Some(u) = parent {
                    u.remove_recursive();
                }
            }
            u.remove_unnecessary_braces();
        }
    }

    pub fn get_or_create_use_tree_list(&self) -> ast::UseTreeList {
        match self.use_tree_list() {
            Some(it) => it,
            None => {
                let position = Position::last_child_of(self.syntax());
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
        ted::insert(Position::first_child_of(self.syntax()), prefix.syntax());
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
                Position::after(last_tree.syntax()),
                vec![
                    make::token(T![,]).into(),
                    make::tokens::single_space().into(),
                    use_tree.syntax.into(),
                ],
            ),
            None => {
                let position = match self.l_curly_token() {
                    Some(l_curly) => Position::after(l_curly),
                    None => Position::last_child_of(self.syntax()),
                };
                (position, vec![use_tree.syntax.into()])
            }
        };
        ted::insert_all_raw(position, elements);
    }
}

impl Removable for ast::Use {
    fn remove(&self) {
        let next_ws = self
            .syntax()
            .next_sibling_or_token()
            .and_then(|it| it.into_token())
            .and_then(ast::Whitespace::cast);
        if let Some(next_ws) = next_ws {
            let ws_text = next_ws.syntax().text();
            if let Some(rest) = ws_text.strip_prefix('\n') {
                if rest.is_empty() {
                    ted::remove(next_ws.syntax());
                } else {
                    ted::replace(next_ws.syntax(), make::tokens::whitespace(rest));
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
                ted::remove(prev_ws.syntax());
            } else {
                ted::replace(prev_ws.syntax(), make::tokens::whitespace(rest));
            }
        }

        ted::remove(self.syntax());
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

impl ast::AssocItemList {
    /// Adds a new associated item after all of the existing associated items.
    ///
    /// Attention! This function does align the first line of `item` with respect to `self`,
    /// but it does _not_ change indentation of other lines (if any).
    pub fn add_item(&self, item: ast::AssocItem) {
        let (indent, position, whitespace) = match self.assoc_items().last() {
            Some(last_item) => (
                IndentLevel::from_node(last_item.syntax()),
                Position::after(last_item.syntax()),
                "\n\n",
            ),
            None => match self.l_curly_token() {
                Some(l_curly) => {
                    normalize_ws_between_braces(self.syntax());
                    (IndentLevel::from_token(&l_curly) + 1, Position::after(&l_curly), "\n")
                }
                None => (IndentLevel::single(), Position::last_child_of(self.syntax()), "\n"),
            },
        };
        let elements: Vec<SyntaxElement> = vec![
            make::tokens::whitespace(&format!("{whitespace}{indent}")).into(),
            item.syntax().clone().into(),
        ];
        ted::insert_all(position, elements);
    }
}

impl ast::RecordExprFieldList {
    pub fn add_field(&self, field: ast::RecordExprField) {
        let is_multiline = self.syntax().text().contains_char('\n');
        let whitespace = if is_multiline {
            let indent = IndentLevel::from_node(self.syntax()) + 1;
            make::tokens::whitespace(&format!("\n{indent}"))
        } else {
            make::tokens::single_space()
        };

        if is_multiline {
            normalize_ws_between_braces(self.syntax());
        }

        let position = match self.fields().last() {
            Some(last_field) => {
                let comma = get_or_insert_comma_after(last_field.syntax());
                Position::after(comma)
            }
            None => match self.l_curly_token() {
                Some(it) => Position::after(it),
                None => Position::last_child_of(self.syntax()),
            },
        };

        ted::insert_all(position, vec![whitespace.into(), field.syntax().clone().into()]);
        if is_multiline {
            ted::insert(Position::after(field.syntax()), ast::make::token(T![,]));
        }
    }
}

impl ast::RecordExprField {
    /// This will either replace the initializer, or in the case that this is a shorthand convert
    /// the initializer into the name ref and insert the expr as the new initializer.
    pub fn replace_expr(&self, expr: ast::Expr) {
        if self.name_ref().is_some() {
            match self.expr() {
                Some(prev) => ted::replace(prev.syntax(), expr.syntax()),
                None => ted::append_child(self.syntax(), expr.syntax()),
            }
            return;
        }
        // this is a shorthand
        if let Some(ast::Expr::PathExpr(path_expr)) = self.expr()
            && let Some(path) = path_expr.path()
            && let Some(name_ref) = path.as_single_name_ref()
        {
            path_expr.syntax().detach();
            let children = vec![
                name_ref.syntax().clone().into(),
                ast::make::token(T![:]).into(),
                ast::make::tokens::single_space().into(),
                expr.syntax().clone().into(),
            ];
            ted::insert_all_raw(Position::last_child_of(self.syntax()), children);
        }
    }
}

impl ast::RecordPatFieldList {
    pub fn add_field(&self, field: ast::RecordPatField) {
        let is_multiline = self.syntax().text().contains_char('\n');
        let whitespace = if is_multiline {
            let indent = IndentLevel::from_node(self.syntax()) + 1;
            make::tokens::whitespace(&format!("\n{indent}"))
        } else {
            make::tokens::single_space()
        };

        if is_multiline {
            normalize_ws_between_braces(self.syntax());
        }

        let position = match self.fields().last() {
            Some(last_field) => {
                let syntax = last_field.syntax();
                let comma = get_or_insert_comma_after(syntax);
                Position::after(comma)
            }
            None => match self.l_curly_token() {
                Some(it) => Position::after(it),
                None => Position::last_child_of(self.syntax()),
            },
        };

        ted::insert_all(position, vec![whitespace.into(), field.syntax().clone().into()]);
        if is_multiline {
            ted::insert(Position::after(field.syntax()), ast::make::token(T![,]));
        }
    }
}

fn get_or_insert_comma_after(syntax: &SyntaxNode) -> SyntaxToken {
    match syntax
        .siblings_with_tokens(Direction::Next)
        .filter_map(|it| it.into_token())
        .find(|it| it.kind() == T![,])
    {
        Some(it) => it,
        None => {
            let comma = ast::make::token(T![,]);
            ted::insert(Position::after(syntax), &comma);
            comma
        }
    }
}

fn normalize_ws_between_braces(node: &SyntaxNode) -> Option<()> {
    let l = node
        .children_with_tokens()
        .filter_map(|it| it.into_token())
        .find(|it| it.kind() == T!['{'])?;
    let r = node
        .children_with_tokens()
        .filter_map(|it| it.into_token())
        .find(|it| it.kind() == T!['}'])?;

    let indent = IndentLevel::from_node(node);

    match l.next_sibling_or_token() {
        Some(ws) if ws.kind() == SyntaxKind::WHITESPACE => {
            if ws.next_sibling_or_token()?.into_token()? == r {
                ted::replace(ws, make::tokens::whitespace(&format!("\n{indent}")));
            }
        }
        Some(ws) if ws.kind() == T!['}'] => {
            ted::insert(Position::after(l), make::tokens::whitespace(&format!("\n{indent}")));
        }
        _ => (),
    }
    Some(())
}

impl ast::IdentPat {
    pub fn set_pat(&self, pat: Option<ast::Pat>) {
        match pat {
            None => {
                if let Some(at_token) = self.at_token() {
                    // Remove `@ Pat`
                    let start = at_token.clone().into();
                    let end = self
                        .pat()
                        .map(|it| it.syntax().clone().into())
                        .unwrap_or_else(|| at_token.into());

                    ted::remove_all(start..=end);

                    // Remove any trailing ws
                    if let Some(last) =
                        self.syntax().last_token().filter(|it| it.kind() == WHITESPACE)
                    {
                        last.detach();
                    }
                }
            }
            Some(pat) => {
                if let Some(old_pat) = self.pat() {
                    // Replace existing pattern
                    ted::replace(old_pat.syntax(), pat.syntax())
                } else if let Some(at_token) = self.at_token() {
                    // Have an `@` token but not a pattern yet
                    ted::insert(ted::Position::after(at_token), pat.syntax());
                } else {
                    // Don't have an `@`, should have a name
                    let name = self.name().unwrap();

                    ted::insert_all(
                        ted::Position::after(name.syntax()),
                        vec![
                            make::token(T![@]).into(),
                            make::tokens::single_space().into(),
                            pat.syntax().clone().into(),
                        ],
                    )
                }
            }
        }
    }
}

pub trait HasVisibilityEdit: ast::HasVisibility {
    fn set_visibility(&self, visibility: Option<ast::Visibility>) {
        if let Some(visibility) = visibility {
            match self.visibility() {
                Some(current_visibility) => {
                    ted::replace(current_visibility.syntax(), visibility.syntax())
                }
                None => {
                    let vis_before = self
                        .syntax()
                        .children_with_tokens()
                        .find(|it| !matches!(it.kind(), WHITESPACE | COMMENT | ATTR))
                        .unwrap_or_else(|| self.syntax().first_child_or_token().unwrap());

                    ted::insert(ted::Position::before(vis_before), visibility.syntax());
                }
            }
        } else if let Some(visibility) = self.visibility() {
            ted::remove(visibility.syntax());
        }
    }
}

impl<T: ast::HasVisibility> HasVisibilityEdit for T {}

pub trait Indent: AstNode + Clone + Sized {
    fn indent_level(&self) -> IndentLevel {
        IndentLevel::from_node(self.syntax())
    }
    fn indent(&self, by: IndentLevel) {
        by.increase_indent(self.syntax());
    }
    fn dedent(&self, by: IndentLevel) {
        by.decrease_indent(self.syntax());
    }
    fn reindent_to(&self, target_level: IndentLevel) {
        let current_level = IndentLevel::from_node(self.syntax());
        self.dedent(current_level);
        self.indent(target_level);
    }
}

impl<N: AstNode + Clone> Indent for N {}

#[cfg(test)]
mod tests {
    use std::fmt;

    use parser::Edition;

    use crate::SourceFile;

    use super::*;

    fn ast_mut_from_text<N: AstNode>(text: &str) -> N {
        let parse = SourceFile::parse(text, Edition::CURRENT);
        parse.tree().syntax().descendants().find_map(N::cast).unwrap().clone_for_update()
    }

    #[test]
    fn test_create_generic_param_list() {
        fn check_create_gpl<N: GenericParamsOwnerEdit + fmt::Display>(before: &str, after: &str) {
            let gpl_owner = ast_mut_from_text::<N>(before);
            gpl_owner.get_or_create_generic_param_list();
            assert_eq!(gpl_owner.to_string(), after);
        }

        check_create_gpl::<ast::Fn>("fn foo", "fn foo<>");
        check_create_gpl::<ast::Fn>("fn foo() {}", "fn foo<>() {}");

        check_create_gpl::<ast::Impl>("impl", "impl<>");
        check_create_gpl::<ast::Impl>("impl Struct {}", "impl<> Struct {}");
        check_create_gpl::<ast::Impl>("impl Trait for Struct {}", "impl<> Trait for Struct {}");

        check_create_gpl::<ast::Trait>("trait Trait<>", "trait Trait<>");
        check_create_gpl::<ast::Trait>("trait Trait<> {}", "trait Trait<> {}");

        check_create_gpl::<ast::Struct>("struct A", "struct A<>");
        check_create_gpl::<ast::Struct>("struct A;", "struct A<>;");
        check_create_gpl::<ast::Struct>("struct A();", "struct A<>();");
        check_create_gpl::<ast::Struct>("struct A {}", "struct A<> {}");

        check_create_gpl::<ast::Enum>("enum E", "enum E<>");
        check_create_gpl::<ast::Enum>("enum E {", "enum E<> {");
    }

    #[test]
    fn test_increase_indent() {
        let arm_list = ast_mut_from_text::<ast::Fn>(
            "fn foo() {
    ;
    ;
}",
        );
        arm_list.indent(IndentLevel(2));
        assert_eq!(
            arm_list.to_string(),
            "fn foo() {
            ;
            ;
        }",
        );
    }

    #[test]
    fn test_ident_pat_set_pat() {
        #[track_caller]
        fn check(before: &str, expected: &str, pat: Option<ast::Pat>) {
            let pat = pat.map(|it| it.clone_for_update());

            let ident_pat = ast_mut_from_text::<ast::IdentPat>(&format!("fn f() {{ {before} }}"));
            ident_pat.set_pat(pat);

            let after = ast_mut_from_text::<ast::IdentPat>(&format!("fn f() {{ {expected} }}"));
            assert_eq!(ident_pat.to_string(), after.to_string());
        }

        // replacing
        check("let a @ _;", "let a @ ();", Some(make::tuple_pat([]).into()));

        // note: no trailing semicolon is added for the below tests since it
        // seems to be picked up by the ident pat during error recovery?

        // adding
        check("let a ", "let a @ ()", Some(make::tuple_pat([]).into()));
        check("let a @ ", "let a @ ()", Some(make::tuple_pat([]).into()));

        // removing
        check("let a @ ()", "let a", None);
        check("let a @ ", "let a", None);
    }
}
