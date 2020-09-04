use syntax::ast::{self, edit::AstNodeEdit, make, AstNode, GenericParamsOwner};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_impl_trait_with_generic
//
// Replaces `impl Trait` function argument with the named generic.
//
// ```
// fn foo(bar: <|>impl Bar) {}
// ```
// ->
// ```
// fn foo<B: Bar>(bar: B) {}
// ```
pub(crate) fn replace_impl_trait_with_generic(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let type_impl_trait = ctx.find_node_at_offset::<ast::ImplTraitType>()?;
    let type_param = type_impl_trait.syntax().parent().and_then(ast::Param::cast)?;
    let type_fn = type_param.syntax().ancestors().find_map(ast::Fn::cast)?;

    let impl_trait_ty = type_impl_trait.type_bound_list()?;

    let target = type_fn.syntax().text_range();
    acc.add(
        AssistId("replace_impl_trait_with_generic", AssistKind::RefactorRewrite),
        "Replace impl trait with generic",
        target,
        |edit| {
            let generic_letter = impl_trait_ty.to_string().chars().next().unwrap().to_string();

            let generic_param_list = type_fn
                .generic_param_list()
                .unwrap_or_else(|| make::generic_param_list(None))
                .append_param(make::generic_param(generic_letter.clone(), Some(impl_trait_ty)));

            let new_type_fn = type_fn
                .replace_descendant::<ast::Type>(type_impl_trait.into(), make::ty(&generic_letter))
                .with_generic_param_list(generic_param_list);

            edit.replace_ast(type_fn.clone(), new_type_fn);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::check_assist;

    #[test]
    fn replace_impl_trait_with_generic_params() {
        check_assist(
            replace_impl_trait_with_generic,
            r#"
            fn foo<G>(bar: <|>impl Bar) {}
            "#,
            r#"
            fn foo<G, B: Bar>(bar: B) {}
            "#,
        );
    }

    #[test]
    fn replace_impl_trait_without_generic_params() {
        check_assist(
            replace_impl_trait_with_generic,
            r#"
            fn foo(bar: <|>impl Bar) {}
            "#,
            r#"
            fn foo<B: Bar>(bar: B) {}
            "#,
        );
    }

    #[test]
    fn replace_two_impl_trait_with_generic_params() {
        check_assist(
            replace_impl_trait_with_generic,
            r#"
            fn foo<G>(foo: impl Foo, bar: <|>impl Bar) {}
            "#,
            r#"
            fn foo<G, B: Bar>(foo: impl Foo, bar: B) {}
            "#,
        );
    }

    #[test]
    fn replace_impl_trait_with_empty_generic_params() {
        check_assist(
            replace_impl_trait_with_generic,
            r#"
            fn foo<>(bar: <|>impl Bar) {}
            "#,
            r#"
            fn foo<B: Bar>(bar: B) {}
            "#,
        );
    }

    #[test]
    fn replace_impl_trait_with_empty_multiline_generic_params() {
        check_assist(
            replace_impl_trait_with_generic,
            r#"
            fn foo<
            >(bar: <|>impl Bar) {}
            "#,
            r#"
            fn foo<B: Bar
            >(bar: B) {}
            "#,
        );
    }

    #[test]
    #[ignore = "This case is very rare but there is no simple solutions to fix it."]
    fn replace_impl_trait_with_exist_generic_letter() {
        check_assist(
            replace_impl_trait_with_generic,
            r#"
            fn foo<B>(bar: <|>impl Bar) {}
            "#,
            r#"
            fn foo<B, C: Bar>(bar: C) {}
            "#,
        );
    }

    #[test]
    fn replace_impl_trait_with_multiline_generic_params() {
        check_assist(
            replace_impl_trait_with_generic,
            r#"
            fn foo<
                G: Foo,
                F,
                H,
            >(bar: <|>impl Bar) {}
            "#,
            r#"
            fn foo<
                G: Foo,
                F,
                H, B: Bar
            >(bar: B) {}
            "#,
        );
    }

    #[test]
    fn replace_impl_trait_multiple() {
        check_assist(
            replace_impl_trait_with_generic,
            r#"
            fn foo(bar: <|>impl Foo + Bar) {}
            "#,
            r#"
            fn foo<F: Foo + Bar>(bar: F) {}
            "#,
        );
    }
}
