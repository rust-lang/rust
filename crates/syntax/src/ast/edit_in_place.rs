//! Structural editing for ast.

use std::iter::empty;

use parser::{SyntaxKind, T};
use rowan::SyntaxElement;

use crate::{
    algo::neighbor,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make, GenericParamsOwner,
    },
    ted::{self, Position},
    AstNode, AstToken, Direction,
    SyntaxKind::{ATTR, COMMENT, WHITESPACE},
    SyntaxNode,
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

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
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

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
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

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
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

    fn get_or_create_where_clause(&self) -> ast::WhereClause {
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

pub trait AttrsOwnerEdit: ast::AttrsOwner + AstNodeEdit {
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
}

impl<T: ast::AttrsOwner + AstNodeEdit> AttrsOwnerEdit for T {}

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

impl ast::PathSegment {
    pub fn get_or_create_generic_arg_list(&self) -> ast::GenericArgList {
        if self.generic_arg_list().is_none() {
            let arg_list = make::generic_arg_list().clone_for_update();
            ted::append_child(self.syntax(), arg_list.syntax())
        }
        self.generic_arg_list().unwrap()
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

impl ast::Impl {
    pub fn get_or_create_assoc_item_list(&self) -> ast::AssocItemList {
        if self.assoc_item_list().is_none() {
            let assoc_item_list = make::assoc_item_list().clone_for_update();
            ted::append_child(self.syntax(), assoc_item_list.syntax());
        }
        self.assoc_item_list().unwrap()
    }
}

impl ast::AssocItemList {
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
        let elements: Vec<SyntaxElement<_>> = vec![
            make::tokens::whitespace(&format!("{}{}", whitespace, indent)).into(),
            item.syntax().clone().into(),
        ];
        ted::insert_all(position, elements);
    }
}

impl ast::Fn {
    pub fn get_or_create_body(&self) -> ast::BlockExpr {
        if self.body().is_none() {
            let body = make::ext::empty_block_expr().clone_for_update();
            match self.semicolon_token() {
                Some(semi) => {
                    ted::replace(semi, body.syntax());
                    ted::insert(Position::before(body.syntax), make::tokens::single_space());
                }
                None => ted::append_child(self.syntax(), body.syntax()),
            }
        }
        self.body().unwrap()
    }
}

impl ast::MatchArm {
    pub fn remove(&self) {
        if let Some(sibling) = self.syntax().prev_sibling_or_token() {
            if sibling.kind() == SyntaxKind::WHITESPACE {
                ted::remove(sibling);
            }
        }
        if let Some(sibling) = self.syntax().next_sibling_or_token() {
            if sibling.kind() == T![,] {
                ted::remove(sibling);
            }
        }
        ted::remove(self.syntax());
    }
}

impl ast::MatchArmList {
    pub fn add_arm(&self, arm: ast::MatchArm) {
        normalize_ws_between_braces(self.syntax());
        let mut elements = Vec::new();
        let position = match self.arms().last() {
            Some(last_arm) => {
                if needs_comma(&last_arm) {
                    ted::append_child(last_arm.syntax(), make::token(SyntaxKind::COMMA));
                }
                Position::after(last_arm.syntax().clone())
            }
            None => match self.l_curly_token() {
                Some(it) => Position::after(it),
                None => Position::last_child_of(self.syntax()),
            },
        };
        let indent = IndentLevel::from_node(self.syntax()) + 1;
        elements.push(make::tokens::whitespace(&format!("\n{}", indent)).into());
        elements.push(arm.syntax().clone().into());
        if needs_comma(&arm) {
            ted::append_child(arm.syntax(), make::token(SyntaxKind::COMMA));
        }
        ted::insert_all(position, elements);

        fn needs_comma(arm: &ast::MatchArm) -> bool {
            arm.expr().map_or(false, |e| !e.is_block_like()) && arm.comma_token().is_none()
        }
    }
}

impl ast::RecordExprFieldList {
    pub fn add_field(&self, field: ast::RecordExprField) {
        let is_multiline = self.syntax().text().contains_char('\n');
        let whitespace = if is_multiline {
            let indent = IndentLevel::from_node(self.syntax()) + 1;
            make::tokens::whitespace(&format!("\n{}", indent))
        } else {
            make::tokens::single_space()
        };

        if is_multiline {
            normalize_ws_between_braces(self.syntax());
        }

        let position = match self.fields().last() {
            Some(last_field) => {
                let comma = match last_field
                    .syntax()
                    .siblings_with_tokens(Direction::Next)
                    .filter_map(|it| it.into_token())
                    .find(|it| it.kind() == T![,])
                {
                    Some(it) => it,
                    None => {
                        let comma = ast::make::token(T![,]);
                        ted::insert(Position::after(last_field.syntax()), &comma);
                        comma
                    }
                };
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

impl ast::BlockExpr {
    pub fn push_front(&self, statement: ast::Stmt) {
        ted::insert(Position::after(self.l_curly_token().unwrap()), statement.syntax());
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
                ted::replace(ws, make::tokens::whitespace(&format!("\n{}", indent)));
            }
        }
        Some(ws) if ws.kind() == T!['}'] => {
            ted::insert(Position::after(l), make::tokens::whitespace(&format!("\n{}", indent)));
        }
        _ => (),
    }
    Some(())
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
