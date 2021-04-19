//! Structural editing for ast.

use std::iter::empty;

use parser::T;

use crate::{
    algo::neighbor,
    ast::{self, edit::AstNodeEdit, make, GenericParamsOwner, WhereClause},
    ted::{self, Position},
    AstNode, AstToken, Direction,
};

use super::NameOwner;

pub trait GenericParamsOwnerEdit: ast::GenericParamsOwner + AstNodeEdit {
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

    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(ty) = self.ret_type() {
                Position::after(ty.syntax())
            } else if let Some(param_list) = self.param_list() {
                Position::after(param_list.syntax())
            } else {
                Position::last_child_of(self.syntax())
            };
            create_where_clause(position)
        }
        self.where_clause().unwrap()
    }
}

impl GenericParamsOwnerEdit for ast::Impl {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(imp_token) = self.impl_token() {
                    Position::after(imp_token)
                } else {
                    Position::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            }
        }
    }

    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(items) = self.assoc_item_list() {
                Position::before(items.syntax())
            } else {
                Position::last_child_of(self.syntax())
            };
            create_where_clause(position)
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

    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(items) = self.assoc_item_list() {
                Position::before(items.syntax())
            } else {
                Position::last_child_of(self.syntax())
            };
            create_where_clause(position)
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

    fn get_or_create_where_clause(&self) -> WhereClause {
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
            create_where_clause(position)
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

    fn get_or_create_where_clause(&self) -> WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(gpl) = self.generic_param_list() {
                Position::after(gpl.syntax())
            } else if let Some(name) = self.name() {
                Position::after(name.syntax())
            } else {
                Position::last_child_of(self.syntax())
            };
            create_where_clause(position)
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

impl ast::GenericParamList {
    pub fn add_generic_param(&self, generic_param: ast::GenericParam) {
        match self.generic_params().last() {
            Some(last_param) => {
                let mut elems = Vec::new();
                if !last_param
                    .syntax()
                    .siblings_with_tokens(Direction::Next)
                    .any(|it| it.kind() == T![,])
                {
                    elems.push(make::token(T![,]).into());
                    elems.push(make::tokens::single_space().into());
                };
                elems.push(generic_param.syntax().clone().into());
                let after_last_param = Position::after(last_param.syntax());
                ted::insert_all(after_last_param, elems);
            }
            None => {
                let after_l_angle = Position::after(self.l_angle_token().unwrap());
                ted::insert(after_l_angle, generic_param.syntax())
            }
        }
    }
}

impl ast::WhereClause {
    pub fn add_predicate(&self, predicate: ast::WherePred) {
        if let Some(pred) = self.predicates().last() {
            if !pred.syntax().siblings_with_tokens(Direction::Next).any(|it| it.kind() == T![,]) {
                ted::append_child_raw(self.syntax(), make::token(T![,]));
            }
        }
        ted::append_child(self.syntax(), predicate.syntax())
    }
}

impl ast::TypeBoundList {
    pub fn remove(&self) {
        if let Some(colon) =
            self.syntax().siblings_with_tokens(Direction::Prev).find(|it| it.kind() == T![:])
        {
            ted::remove_all(colon..=self.syntax().clone().into())
        } else {
            ted::remove(self.syntax())
        }
    }
}

impl ast::UseTree {
    pub fn remove(&self) {
        for &dir in [Direction::Next, Direction::Prev].iter() {
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
        ted::remove(self.syntax())
    }
}

impl ast::Use {
    pub fn remove(&self) {
        let next_ws = self
            .syntax()
            .next_sibling_or_token()
            .and_then(|it| it.into_token())
            .and_then(ast::Whitespace::cast);
        if let Some(next_ws) = next_ws {
            let ws_text = next_ws.syntax().text();
            if let Some(rest) = ws_text.strip_prefix('\n') {
                if rest.is_empty() {
                    ted::remove(next_ws.syntax())
                } else {
                    ted::replace(next_ws.syntax(), make::tokens::whitespace(rest))
                }
            }
        }
        ted::remove(self.syntax())
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use crate::SourceFile;

    use super::*;

    fn ast_mut_from_text<N: AstNode>(text: &str) -> N {
        let parse = SourceFile::parse(text);
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
}
