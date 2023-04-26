use syntax::{
    ast::{
        self,
        make::{self, impl_trait_type},
        HasGenericParams, HasName, HasTypeBounds,
    },
    ted, AstNode,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_named_generic_with_impl
//
// Replaces named generic with an `impl Trait` in function argument.
//
// ```
// fn new<P$0: AsRef<Path>>(location: P) -> Self {}
// ```
// ->
// ```
// fn new(location: impl AsRef<Path>) -> Self {}
// ```
pub(crate) fn replace_named_generic_with_impl(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    // finds `<P: AsRef<Path>>`
    let type_param = ctx.find_node_at_offset::<ast::TypeParam>()?;

    // The list of type bounds / traits for generic name `P`
    let type_bound_list = type_param.type_bound_list()?;

    // returns `P`
    let type_param_name = type_param.name()?;

    let fn_ = type_param.syntax().ancestors().find_map(ast::Fn::cast)?;
    let params = fn_
        .param_list()?
        .params()
        .filter_map(|param| {
            // function parameter type needs to match generic type name
            if let ast::Type::PathType(path_type) = param.ty()? {
                let left = path_type.path()?.segment()?.name_ref()?.ident_token()?.to_string();
                let right = type_param_name.to_string();
                if left == right {
                    Some(param)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if params.is_empty() {
        return None;
    }

    let target = type_param.syntax().text_range();

    acc.add(
        AssistId("replace_named_generic_with_impl", AssistKind::RefactorRewrite),
        "Replace named generic with impl",
        target,
        |edit| {
            let type_param = edit.make_mut(type_param);
            let fn_ = edit.make_mut(fn_);

            // Replace generic type in `<P: AsRef<Path>>` to `<P>`
            let new_ty = make::ty(&type_param_name.to_string()).clone_for_update();
            ted::replace(type_param.syntax(), new_ty.syntax());

            if let Some(generic_params) = fn_.generic_param_list() {
                if generic_params.generic_params().count() == 0 {
                    ted::remove(generic_params.syntax());
                }
            }

            // Replace generic type parameter: `foo(p: P)` -> `foo(p: impl AsRef<Path>)`
            let new_bounds = impl_trait_type(type_bound_list).clone_for_update();

            for param in params {
                if let Some(ast::Type::PathType(param_type)) = param.ty() {
                    let param_type = edit.make_mut(param_type).clone_for_update();
                    ted::replace(param_type.syntax(), new_bounds.syntax());
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::check_assist;

    #[test]
    fn replace_generic_moves_into_function() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<T$0: ToString>(input: T) -> Self {}"#,
            r#"fn new(input: impl ToString) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_with_inner_associated_type() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<P$0: AsRef<Path>>(input: P) -> Self {}"#,
            r#"fn new(input: impl AsRef<Path>) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_trait_applies_to_all_matching_params() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<T$0: ToString>(a: T, b: T) -> Self {}"#,
            r#"fn new(a: impl ToString, b: impl ToString) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_with_multiple_generic_names() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<P: AsRef<Path>, T$0: ToString>(t: T, p: P) -> Self {}"#,
            r#"fn new<P: AsRef<Path>>(t: impl ToString, p: P) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_with_multiple_trait_bounds() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<P$0: Send + Sync>(p: P) -> Self {}"#,
            r#"fn new(p: impl Send + Sync) -> Self {}"#,
        );
    }
}
