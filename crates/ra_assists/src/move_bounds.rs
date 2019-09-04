use hir::db::HirDatabase;
use ra_syntax::{
    ast::{self, AstNode, NameOwner, TypeBoundsOwner},
    SyntaxElement,
    SyntaxKind::*,
    TextRange,
};

use crate::{ast_editor::AstBuilder, Assist, AssistCtx, AssistId};

pub(crate) fn move_bounds_to_where_clause(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let type_param_list = ctx.node_at_offset::<ast::TypeParamList>()?;

    let mut type_params = type_param_list.type_params();
    if type_params.all(|p| p.type_bound_list().is_none()) {
        return None;
    }

    let parent = type_param_list.syntax().parent()?;
    if parent.children_with_tokens().find(|it| it.kind() == WHERE_CLAUSE).is_some() {
        return None;
    }

    let anchor: SyntaxElement = match parent.kind() {
        FN_DEF => ast::FnDef::cast(parent)?.body()?.syntax().clone().into(),
        TRAIT_DEF => ast::TraitDef::cast(parent)?.item_list()?.syntax().clone().into(),
        IMPL_BLOCK => ast::ImplBlock::cast(parent)?.item_list()?.syntax().clone().into(),
        ENUM_DEF => ast::EnumDef::cast(parent)?.variant_list()?.syntax().clone().into(),
        STRUCT_DEF => parent
            .children_with_tokens()
            .find(|it| it.kind() == RECORD_FIELD_DEF_LIST || it.kind() == SEMI)?,
        _ => return None,
    };

    ctx.add_action(
        AssistId("move_bounds_to_where_clause"),
        "move_bounds_to_where_clause",
        |edit| {
            let type_params = type_param_list.type_params().collect::<Vec<_>>();

            for param in &type_params {
                if let Some(bounds) = param.type_bound_list() {
                    let colon = param
                        .syntax()
                        .children_with_tokens()
                        .find(|it| it.kind() == COLON)
                        .unwrap();
                    let start = colon.text_range().start();
                    let end = bounds.syntax().text_range().end();
                    edit.delete(TextRange::from_to(start, end));
                }
            }

            let predicates = type_params.iter().filter_map(build_predicate);
            let where_clause = AstBuilder::<ast::WhereClause>::from_predicates(predicates);

            let to_insert = match anchor.prev_sibling_or_token() {
                Some(ref elem) if elem.kind() == WHITESPACE => {
                    format!("{} ", where_clause.syntax())
                }
                _ => format!(" {}", where_clause.syntax()),
            };
            edit.insert(anchor.text_range().start(), to_insert);
            edit.target(type_param_list.syntax().text_range());
        },
    );

    ctx.build()
}

fn build_predicate(param: &ast::TypeParam) -> Option<ast::WherePred> {
    let path = AstBuilder::<ast::Path>::from_name(param.name()?);
    let predicate =
        AstBuilder::<ast::WherePred>::from_pieces(path, param.type_bound_list()?.bounds());
    Some(predicate)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::helpers::check_assist;

    #[test]
    fn move_bounds_to_where_clause_fn() {
        check_assist(
            move_bounds_to_where_clause,
            r#"
            fn foo<T: u32, <|>F: FnOnce(T) -> T>() {}
            "#,
            r#"
            fn foo<T, <|>F>() where T: u32, F: FnOnce(T) -> T {}
            "#,
        );
    }

    #[test]
    fn move_bounds_to_where_clause_impl() {
        check_assist(
            move_bounds_to_where_clause,
            r#"
            impl<U: u32, <|>T> A<U, T> {}
            "#,
            r#"
            impl<U, <|>T> A<U, T> where U: u32 {}
            "#,
        );
    }

    #[test]
    fn move_bounds_to_where_clause_struct() {
        check_assist(
            move_bounds_to_where_clause,
            r#"
            struct A<<|>T: Iterator<Item = u32>> {}
            "#,
            r#"
            struct A<<|>T> where T: Iterator<Item = u32> {}
            "#,
        );
    }

    #[test]
    fn move_bounds_to_where_clause_tuple_struct() {
        check_assist(
            move_bounds_to_where_clause,
            r#"
            struct Pair<<|>T: u32>(T, T);
            "#,
            r#"
            struct Pair<<|>T>(T, T) where T: u32;
            "#,
        );
    }
}
