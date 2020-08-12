use syntax::{
    ast::{self, edit::AstNodeEdit, make, AstNode, NameOwner, TypeBoundsOwner},
    match_ast,
    SyntaxKind::*,
    T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: move_bounds_to_where_clause
//
// Moves inline type bounds to a where clause.
//
// ```
// fn apply<T, U, <|>F: FnOnce(T) -> U>(f: F, x: T) -> U {
//     f(x)
// }
// ```
// ->
// ```
// fn apply<T, U, F>(f: F, x: T) -> U where F: FnOnce(T) -> U {
//     f(x)
// }
// ```
pub(crate) fn move_bounds_to_where_clause(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let type_param_list = ctx.find_node_at_offset::<ast::GenericParamList>()?;

    let mut type_params = type_param_list.type_params();
    if type_params.all(|p| p.type_bound_list().is_none()) {
        return None;
    }

    let parent = type_param_list.syntax().parent()?;
    if parent.children_with_tokens().any(|it| it.kind() == WHERE_CLAUSE) {
        return None;
    }

    let anchor = match_ast! {
        match parent {
            ast::Fn(it) => it.body()?.syntax().clone().into(),
            ast::Trait(it) => it.assoc_item_list()?.syntax().clone().into(),
            ast::Impl(it) => it.assoc_item_list()?.syntax().clone().into(),
            ast::Enum(it) => it.variant_list()?.syntax().clone().into(),
            ast::Struct(it) => {
                it.syntax().children_with_tokens()
                    .find(|it| it.kind() == RECORD_FIELD_LIST || it.kind() == T![;])?
            },
            _ => return None
        }
    };

    let target = type_param_list.syntax().text_range();
    acc.add(
        AssistId("move_bounds_to_where_clause", AssistKind::RefactorRewrite),
        "Move to where clause",
        target,
        |edit| {
            let new_params = type_param_list
                .type_params()
                .filter(|it| it.type_bound_list().is_some())
                .map(|type_param| {
                    let without_bounds = type_param.remove_bounds();
                    (type_param, without_bounds)
                });

            let new_type_param_list = type_param_list.replace_descendants(new_params);
            edit.replace_ast(type_param_list.clone(), new_type_param_list);

            let where_clause = {
                let predicates = type_param_list.type_params().filter_map(build_predicate);
                make::where_clause(predicates)
            };

            let to_insert = match anchor.prev_sibling_or_token() {
                Some(ref elem) if elem.kind() == WHITESPACE => {
                    format!("{} ", where_clause.syntax())
                }
                _ => format!(" {}", where_clause.syntax()),
            };
            edit.insert(anchor.text_range().start(), to_insert);
        },
    )
}

fn build_predicate(param: ast::TypeParam) -> Option<ast::WherePred> {
    let path = {
        let name_ref = make::name_ref(&param.name()?.syntax().to_string());
        let segment = make::path_segment(name_ref);
        make::path_unqualified(segment)
    };
    let predicate = make::where_pred(path, param.type_bound_list()?.bounds());
    Some(predicate)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::check_assist;

    #[test]
    fn move_bounds_to_where_clause_fn() {
        check_assist(
            move_bounds_to_where_clause,
            r#"
            fn foo<T: u32, <|>F: FnOnce(T) -> T>() {}
            "#,
            r#"
            fn foo<T, F>() where T: u32, F: FnOnce(T) -> T {}
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
            impl<U, T> A<U, T> where U: u32 {}
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
            struct A<T> where T: Iterator<Item = u32> {}
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
            struct Pair<T>(T, T) where T: u32;
            "#,
        );
    }
}
