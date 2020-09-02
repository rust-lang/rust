use syntax::ast::{self, edit::AstNodeEdit, make, AstNode, GenericParamsOwner};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_impl_trait_with_generic
//
// Replaces `impl Trait` function argument with the named generic.
pub(crate) fn replace_impl_trait_with_generic(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let type_impl_trait = ctx.find_node_at_offset::<ast::ImplTraitType>()?;
    let type_param = type_impl_trait.syntax().parent().and_then(ast::Param::cast)?;
    let type_fn = type_param.syntax().ancestors().nth(2).and_then(ast::Fn::cast)?;

    let generic_param_list =
        type_fn.generic_param_list().unwrap_or_else(|| make::generic_param_list(None));

    let impl_trait_ty = type_impl_trait
        .syntax()
        .descendants()
        .last()
        .and_then(ast::NameRef::cast)?
        .text()
        .to_string();

    let target = type_fn.syntax().text_range();
    acc.add(
        AssistId("replace_impl_trait_with_generic", AssistKind::RefactorRewrite),
        "Replace impl trait with generic",
        target,
        |edit| {
            let generic_letter = impl_trait_ty[..1].to_string();
            edit.replace_ast::<ast::Type>(type_impl_trait.into(), make::ty(&generic_letter));

            let new_params = generic_param_list
                .append_param(make::generic_param(generic_letter, Some(impl_trait_ty)));
            let new_type_fn = type_fn.replace_descendant(generic_param_list, new_params);
            edit.replace_ast(type_fn.clone(), new_type_fn);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::check_assist;

    #[test]
    fn replace_with_generic_params() {
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
}
