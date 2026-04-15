use either::Either;
use syntax::{
    ast::{self, AstNode, HasName, HasTypeBounds, syntax_factory::SyntaxFactory},
    match_ast,
    syntax_editor::{GetOrCreateWhereClause, Removable},
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
        |builder| {
            let mut edit = builder.make_editor(&parent);

            let new_preds: Vec<ast::WherePred> = type_param_list
                .generic_params()
                .filter_map(|param| build_predicate(param, edit.make()))
                .collect();

            match_ast! {
                match (&parent) {
                    ast::Fn(it) => it.get_or_create_where_clause(&mut edit, new_preds.into_iter()),
                    ast::Trait(it) => it.get_or_create_where_clause(&mut edit, new_preds.into_iter()),
                    ast::Impl(it) => it.get_or_create_where_clause(&mut edit, new_preds.into_iter()),
                    ast::Enum(it) => it.get_or_create_where_clause(&mut edit, new_preds.into_iter()),
                    ast::Struct(it) => it.get_or_create_where_clause(&mut edit, new_preds.into_iter()),
                    ast::TypeAlias(it) => it.get_or_create_where_clause(&mut edit, new_preds.into_iter()),
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
                    tbl.remove(&mut edit);
                }
            }

            builder.add_file_edits(ctx.vfs_file_id(), edit);
        },
    )
}

fn build_predicate(param: ast::GenericParam, make: &SyntaxFactory) -> Option<ast::WherePred> {
    let target = match &param {
        ast::GenericParam::TypeParam(t) => Either::Right(make.ty(&t.name()?.to_string())),
        ast::GenericParam::LifetimeParam(l) => Either::Left(l.lifetime()?),
        ast::GenericParam::ConstParam(_) => return None,
    };
    let predicate = make.where_pred(
        target,
        match param {
            ast::GenericParam::TypeParam(t) => t.type_bound_list()?,
            ast::GenericParam::LifetimeParam(l) => l.type_bound_list()?,
            ast::GenericParam::ConstParam(_) => return None,
        }
        .bounds(),
    );
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
