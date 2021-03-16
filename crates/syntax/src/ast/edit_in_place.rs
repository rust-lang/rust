//! Structural editing for ast.

use std::iter::empty;

use ast::{edit::AstNodeEdit, make, GenericParamsOwner, WhereClause};
use parser::T;

use crate::{
    ast,
    ted::{self, Position},
    AstNode, Direction, SyntaxKind,
};

use super::NameOwner;

pub trait GenericParamsOwnerEdit: ast::GenericParamsOwner + AstNodeEdit {
    fn get_or_create_where_clause(&self) -> ast::WhereClause;
}

impl GenericParamsOwnerEdit for ast::Fn {
    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(ty) = self.ret_type() {
                Position::after(ty.syntax().clone())
            } else if let Some(param_list) = self.param_list() {
                Position::after(param_list.syntax().clone())
            } else {
                Position::last_child_of(self.syntax().clone())
            };
            create_where_clause(position, true)
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Impl {
    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(items) = self.assoc_item_list() {
                Position::before(items.syntax().clone())
            } else {
                Position::last_child_of(self.syntax().clone())
            };
            create_where_clause(position, false)
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Trait {
    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(items) = self.assoc_item_list() {
                Position::before(items.syntax().clone())
            } else {
                Position::last_child_of(self.syntax().clone())
            };
            create_where_clause(position, false)
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Struct {
    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let tfl = self.field_list().and_then(|fl| match fl {
                ast::FieldList::RecordFieldList(_) => None,
                ast::FieldList::TupleFieldList(it) => Some(it),
            });
            let position = if let Some(tfl) = tfl {
                Position::after(tfl.syntax().clone())
            } else if let Some(gpl) = self.generic_param_list() {
                Position::after(gpl.syntax().clone())
            } else if let Some(name) = self.name() {
                Position::after(name.syntax().clone())
            } else {
                Position::last_child_of(self.syntax().clone())
            };
            create_where_clause(position, true)
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Enum {
    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(gpl) = self.generic_param_list() {
                Position::after(gpl.syntax().clone())
            } else if let Some(name) = self.name() {
                Position::after(name.syntax().clone())
            } else {
                Position::last_child_of(self.syntax().clone())
            };
            create_where_clause(position, true)
        }
        self.where_clause().unwrap()
    }
}

fn create_where_clause(position: Position, after: bool) {
    let mut elements = vec![make::where_clause(empty()).clone_for_update().syntax().clone().into()];
    let ws = make::tokens::single_space().into();
    if after {
        elements.insert(0, ws)
    } else {
        elements.push(ws)
    }
    ted::insert_all(position, elements);
}

impl ast::WhereClause {
    pub fn add_predicate(&self, predicate: ast::WherePred) {
        if let Some(pred) = self.predicates().last() {
            if !pred.syntax().siblings_with_tokens(Direction::Next).any(|it| it.kind() == T![,]) {
                ted::append_child(self.syntax().clone(), make::token(T![,]));
            }
        }
        if self.syntax().children_with_tokens().last().map(|it| it.kind())
            != Some(SyntaxKind::WHITESPACE)
        {
            ted::append_child(self.syntax().clone(), make::tokens::single_space());
        }
        ted::append_child(self.syntax().clone(), predicate.syntax().clone())
    }
}

impl ast::TypeBoundList {
    pub fn remove(&self) {
        if let Some(colon) =
            self.syntax().siblings_with_tokens(Direction::Prev).find(|it| it.kind() == T![:])
        {
            ted::remove_all(colon..=self.syntax().clone().into())
        } else {
            ted::remove(self.syntax().clone())
        }
    }
}
