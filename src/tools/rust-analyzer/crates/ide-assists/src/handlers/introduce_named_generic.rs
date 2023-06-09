use syntax::{
    ast::{self, edit_in_place::GenericParamsOwnerEdit, make, AstNode, HasGenericParams},
    ted,
};

use crate::{utils::suggest_name, AssistContext, AssistId, AssistKind, Assists};

// Assist: introduce_named_generic
//
// Replaces `impl Trait` function argument with the named generic.
//
// ```
// fn foo(bar: $0impl Bar) {}
// ```
// ->
// ```
// fn foo<$0B: Bar>(bar: B) {}
// ```
pub(crate) fn introduce_named_generic(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let impl_trait_type = ctx.find_node_at_offset::<ast::ImplTraitType>()?;
    let param = impl_trait_type.syntax().parent().and_then(ast::Param::cast)?;
    let fn_ = param.syntax().ancestors().find_map(ast::Fn::cast)?;

    let type_bound_list = impl_trait_type.type_bound_list()?;

    let target = fn_.syntax().text_range();
    acc.add(
        AssistId("introduce_named_generic", AssistKind::RefactorRewrite),
        "Replace impl trait with generic",
        target,
        |edit| {
            let impl_trait_type = edit.make_mut(impl_trait_type);
            let fn_ = edit.make_mut(fn_);

            let type_param_name = suggest_name::for_generic_parameter(&impl_trait_type);

            let type_param = make::type_param(make::name(&type_param_name), Some(type_bound_list))
                .clone_for_update();
            let new_ty = make::ty(&type_param_name).clone_for_update();

            ted::replace(impl_trait_type.syntax(), new_ty.syntax());
            fn_.get_or_create_generic_param_list().add_generic_param(type_param.into());

            if let Some(cap) = ctx.config.snippet_cap {
                if let Some(generic_param) =
                    fn_.generic_param_list().and_then(|it| it.generic_params().last())
                {
                    edit.add_tabstop_before(cap, generic_param);
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
    fn introduce_named_generic_params() {
        check_assist(
            introduce_named_generic,
            r#"fn foo<G>(bar: $0impl Bar) {}"#,
            r#"fn foo<G, $0B: Bar>(bar: B) {}"#,
        );
    }

    #[test]
    fn replace_impl_trait_without_generic_params() {
        check_assist(
            introduce_named_generic,
            r#"fn foo(bar: $0impl Bar) {}"#,
            r#"fn foo<$0B: Bar>(bar: B) {}"#,
        );
    }

    #[test]
    fn replace_two_impl_trait_with_generic_params() {
        check_assist(
            introduce_named_generic,
            r#"fn foo<G>(foo: impl Foo, bar: $0impl Bar) {}"#,
            r#"fn foo<G, $0B: Bar>(foo: impl Foo, bar: B) {}"#,
        );
    }

    #[test]
    fn replace_impl_trait_with_empty_generic_params() {
        check_assist(
            introduce_named_generic,
            r#"fn foo<>(bar: $0impl Bar) {}"#,
            r#"fn foo<$0B: Bar>(bar: B) {}"#,
        );
    }

    #[test]
    fn replace_impl_trait_with_empty_multiline_generic_params() {
        check_assist(
            introduce_named_generic,
            r#"
fn foo<
>(bar: $0impl Bar) {}
"#,
            r#"
fn foo<$0B: Bar
>(bar: B) {}
"#,
        );
    }

    #[test]
    fn replace_impl_trait_with_exist_generic_letter() {
        // FIXME: This is wrong, we should pick a different name if the one we
        // want is already bound.
        check_assist(
            introduce_named_generic,
            r#"fn foo<B>(bar: $0impl Bar) {}"#,
            r#"fn foo<B, $0B: Bar>(bar: B) {}"#,
        );
    }

    #[test]
    fn replace_impl_trait_with_multiline_generic_params() {
        check_assist(
            introduce_named_generic,
            r#"
fn foo<
    G: Foo,
    F,
    H,
>(bar: $0impl Bar) {}
"#,
            r#"
fn foo<
    G: Foo,
    F,
    H, $0B: Bar,
>(bar: B) {}
"#,
        );
    }

    #[test]
    fn replace_impl_trait_multiple() {
        check_assist(
            introduce_named_generic,
            r#"fn foo(bar: $0impl Foo + Bar) {}"#,
            r#"fn foo<$0F: Foo + Bar>(bar: F) {}"#,
        );
    }
}
