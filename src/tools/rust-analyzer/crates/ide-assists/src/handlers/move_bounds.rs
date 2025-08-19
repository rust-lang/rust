use either::Either;
use syntax::{
    ast::{
        self, AstNode, HasName, HasTypeBounds,
        edit_in_place::{GenericParamsOwnerEdit, Removable},
        make,
    },
    match_ast,
};

use crate::{AssistContext, AssistId, Assists};

// Assist: move_bounds_to_where_clause
//
// Moves inline type bounds to a where clause.
//
// ```
// fn apply<T, U, $0F: FnOnce(T) -> U>(f: F, x: T) -> U {
//     f(x)
// }
// ```
// ->
// ```
// fn apply<T, U, F>(f: F, x: T) -> U where F: FnOnce(T) -> U {
//     f(x)
// }
// ```
pub(crate) fn move_bounds_to_where_clause(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let type_param_list = ctx.find_node_at_offset::<ast::GenericParamList>()?;

    let mut type_params = type_param_list.generic_params();
    if type_params.all(|p| match p {
        ast::GenericParam::TypeParam(t) => t.type_bound_list().is_none(),
        ast::GenericParam::LifetimeParam(l) => l.type_bound_list().is_none(),
        ast::GenericParam::ConstParam(_) => true,
    }) {
        return None;
    }

    let parent = type_param_list.syntax().parent()?;

    let target = type_param_list.syntax().text_range();
    acc.add(
        AssistId::refactor_rewrite("move_bounds_to_where_clause"),
        "Move to where clause",
        target,
        |edit| {
            let type_param_list = edit.make_mut(type_param_list);
            let parent = edit.make_syntax_mut(parent);

            let where_clause: ast::WhereClause = match_ast! {
                match parent {
                    ast::Fn(it) => it.get_or_create_where_clause(),
                    ast::Trait(it) => it.get_or_create_where_clause(),
                    ast::TraitAlias(it) => it.get_or_create_where_clause(),
                    ast::Impl(it) => it.get_or_create_where_clause(),
                    ast::Enum(it) => it.get_or_create_where_clause(),
                    ast::Struct(it) => it.get_or_create_where_clause(),
                    ast::TypeAlias(it) => it.get_or_create_where_clause(),
                    _ => return,
                }
            };

            for generic_param in type_param_list.generic_params() {
                let param: &dyn HasTypeBounds = match &generic_param {
                    ast::GenericParam::TypeParam(t) => t,
                    ast::GenericParam::LifetimeParam(l) => l,
                    ast::GenericParam::ConstParam(_) => continue,
                };
                if let Some(tbl) = param.type_bound_list() {
                    if let Some(predicate) = build_predicate(generic_param) {
                        where_clause.add_predicate(predicate)
                    }
                    tbl.remove()
                }
            }
        },
    )
}

fn build_predicate(param: ast::GenericParam) -> Option<ast::WherePred> {
    let target = match &param {
        ast::GenericParam::TypeParam(t) => {
            Either::Right(make::ty_path(make::ext::ident_path(&t.name()?.to_string())))
        }
        ast::GenericParam::LifetimeParam(l) => Either::Left(l.lifetime()?),
        ast::GenericParam::ConstParam(_) => return None,
    };
    let predicate = make::where_pred(
        target,
        match param {
            ast::GenericParam::TypeParam(t) => t.type_bound_list()?,
            ast::GenericParam::LifetimeParam(l) => l.type_bound_list()?,
            ast::GenericParam::ConstParam(_) => return None,
        }
        .bounds(),
    );
    Some(predicate.clone_for_update())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::check_assist;

    #[test]
    fn move_bounds_to_where_clause_fn() {
        check_assist(
            move_bounds_to_where_clause,
            r#"fn foo<T: u32, $0F: FnOnce(T) -> T>() {}"#,
            r#"fn foo<T, F>() where T: u32, F: FnOnce(T) -> T {}"#,
        );
    }

    #[test]
    fn move_bounds_to_where_clause_impl() {
        check_assist(
            move_bounds_to_where_clause,
            r#"impl<U: u32, $0T> A<U, T> {}"#,
            r#"impl<U, T> A<U, T> where U: u32 {}"#,
        );
    }

    #[test]
    fn move_bounds_to_where_clause_struct() {
        check_assist(
            move_bounds_to_where_clause,
            r#"struct A<$0T: Iterator<Item = u32>> {}"#,
            r#"struct A<T> where T: Iterator<Item = u32> {}"#,
        );
    }

    #[test]
    fn move_bounds_to_where_clause_tuple_struct() {
        check_assist(
            move_bounds_to_where_clause,
            r#"struct Pair<$0T: u32>(T, T);"#,
            r#"struct Pair<T>(T, T) where T: u32;"#,
        );
    }

    #[test]
    fn move_bounds_to_where_clause_trait() {
        check_assist(
            move_bounds_to_where_clause,
            r#"trait T<'a: 'static, $0T: u32> {}"#,
            r#"trait T<'a, T> where 'a: 'static, T: u32 {}"#,
        );
    }
}
