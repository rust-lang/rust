use ide_db::{
    assists::{AssistId, AssistKind},
    famous_defs::FamousDefs,
};
use syntax::{
    ast::{self, HasVisibility},
    AstNode, NodeOrToken, SyntaxKind, SyntaxNode, SyntaxToken, TextRange,
};

use crate::{AssistContext, Assists};

// Assist: toggle_async_sugar
//
// Rewrites asynchronous function into `impl Future` and back.
// This action does not touch the function body and therefore `async { 0 }`
// block does not transform to just `0`.
//
// ```
// pub async f$0n foo() -> usize {
//     0
// }
// ```
// ->
// ```
// pub fn foo() -> impl Future<Output = usize> {
//     0
// }
// ```
pub(crate) fn toggle_async_sugar(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let function: ast::Fn = ctx.find_node_at_offset()?;
    match (function.async_token(), function.ret_type()) {
        // async function returning futures cannot be flattened
        // const async is not yet supported
        (None, Some(ret_type)) if function.const_token().is_none() => {
            add_async(acc, ctx, function, ret_type)
        }
        (Some(async_token), ret_type) => remove_async(function, ret_type, acc, async_token),
        _ => None,
    }
}

fn add_async(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    function: ast::Fn,
    ret_type: ast::RetType,
) -> Option<()> {
    let ast::Type::ImplTraitType(return_impl_trait) = ret_type.ty()? else {
        return None;
    };

    let main_trait_path = return_impl_trait
        .type_bound_list()?
        .bounds()
        .filter_map(|bound| match bound.ty() {
            Some(ast::Type::PathType(trait_path)) => trait_path.path(),
            _ => None,
        })
        .next()?;

    let trait_type = ctx.sema.resolve_trait(&main_trait_path)?;
    let scope = ctx.sema.scope(main_trait_path.syntax())?;
    if trait_type != FamousDefs(&ctx.sema, scope.krate()).core_future_Future()? {
        return None;
    }
    let future_output = unwrap_future_output(main_trait_path)?;

    acc.add(
        AssistId("toggle_async_sugar", AssistKind::RefactorRewrite),
        "Convert `impl Future` into async",
        function.syntax().text_range(),
        |builder| {
            match future_output {
                ast::Type::TupleType(_) => {
                    let mut ret_type_range = ret_type.syntax().text_range();

                    // find leftover whitespace
                    let whitespace_range = function
                        .param_list()
                        .as_ref()
                        .map(|params| NodeOrToken::Node(params.syntax()))
                        .and_then(following_whitespace);

                    if let Some(whitespace_range) = whitespace_range {
                        ret_type_range =
                            TextRange::new(whitespace_range.start(), ret_type_range.end());
                    }

                    builder.delete(ret_type_range);
                }
                _ => {
                    builder.replace(
                        return_impl_trait.syntax().text_range(),
                        future_output.syntax().text(),
                    );
                }
            }

            let (place_for_async, async_kw) = match function.visibility() {
                Some(vis) => (vis.syntax().text_range().end(), " async"),
                None => (function.syntax().text_range().start(), "async "),
            };
            builder.insert(place_for_async, async_kw);
        },
    )
}

fn remove_async(
    function: ast::Fn,
    ret_type: Option<ast::RetType>,
    acc: &mut Assists,
    async_token: SyntaxToken,
) -> Option<()> {
    let rparen = function.param_list()?.r_paren_token()?;
    let return_type = match ret_type {
        // unable to get a `ty` makes the action unapplicable
        Some(ret_type) => Some(ret_type.ty()?),
        // No type means `-> ()`
        None => None,
    };

    acc.add(
        AssistId("toggle_async_sugar", AssistKind::RefactorRewrite),
        "Convert async into `impl Future`",
        function.syntax().text_range(),
        |builder| {
            let mut async_range = async_token.text_range();

            if let Some(whitespace_range) = following_whitespace(NodeOrToken::Token(async_token)) {
                async_range = TextRange::new(async_range.start(), whitespace_range.end());
            }
            builder.delete(async_range);

            match return_type {
                Some(ret_type) => builder.replace(
                    ret_type.syntax().text_range(),
                    format!("impl Future<Output = {ret_type}>"),
                ),
                None => builder.insert(rparen.text_range().end(), " -> impl Future<Output = ()>"),
            }
        },
    )
}

fn unwrap_future_output(path: ast::Path) -> Option<ast::Type> {
    let future_trait = path.segments().last()?;
    let assoc_list = future_trait.generic_arg_list()?;
    let future_assoc = assoc_list.generic_args().next()?;
    match future_assoc {
        ast::GenericArg::AssocTypeArg(output_type) => output_type.ty(),
        _ => None,
    }
}

fn following_whitespace(nt: NodeOrToken<&SyntaxNode, SyntaxToken>) -> Option<TextRange> {
    let next_token = match nt {
        NodeOrToken::Node(node) => node.next_sibling_or_token(),
        NodeOrToken::Token(token) => token.next_sibling_or_token(),
    }?;
    (next_token.kind() == SyntaxKind::WHITESPACE).then_some(next_token.text_range())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn sugar_with_use() {
        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    use core::future::Future;
    f$0n foo() -> impl Future<Output = ()> {
        todo!()
    }
    "#,
            r#"
    use core::future::Future;
    async fn foo() {
        todo!()
    }
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    use core::future::Future;
    f$0n foo() -> impl Future<Output = usize> {
        todo!()
    }
    "#,
            r#"
    use core::future::Future;
    async fn foo() -> usize {
        todo!()
    }
    "#,
        );
    }

    #[test]
    fn desugar_with_use() {
        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    use core::future::Future;
    async f$0n foo() {
        todo!()
    }
    "#,
            r#"
    use core::future::Future;
    fn foo() -> impl Future<Output = ()> {
        todo!()
    }
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    use core::future::Future;
    async f$0n foo() -> usize {
        todo!()
    }
    "#,
            r#"
    use core::future::Future;
    fn foo() -> impl Future<Output = usize> {
        todo!()
    }
    "#,
        );
    }

    #[test]
    fn sugar_without_use() {
        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    f$0n foo() -> impl core::future::Future<Output = ()> {
        todo!()
    }
    "#,
            r#"
    async fn foo() {
        todo!()
    }
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    f$0n foo() -> impl core::future::Future<Output = usize> {
        todo!()
    }
    "#,
            r#"
    async fn foo() -> usize {
        todo!()
    }
    "#,
        );
    }

    #[test]
    fn desugar_without_use() {
        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    async f$0n foo() {
        todo!()
    }
    "#,
            r#"
    fn foo() -> impl Future<Output = ()> {
        todo!()
    }
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    async f$0n foo() -> usize {
        todo!()
    }
    "#,
            r#"
    fn foo() -> impl Future<Output = usize> {
        todo!()
    }
    "#,
        );
    }

    #[test]
    fn sugar_not_applicable() {
        check_assist_not_applicable(
            toggle_async_sugar,
            r#"
    //- minicore: future
    trait Future {
        type Output;
    }
    f$0n foo() -> impl Future<Output = ()> {
        todo!()
    }
    "#,
        );

        check_assist_not_applicable(
            toggle_async_sugar,
            r#"
    //- minicore: future
    trait Future {
        type Output;
    }
    f$0n foo() -> impl Future<Output = usize> {
        todo!()
    }
    "#,
        );
    }

    #[test]
    fn sugar_definition_with_use() {
        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    use core::future::Future;
    f$0n foo() -> impl Future<Output = ()>;
    "#,
            r#"
    use core::future::Future;
    async fn foo();
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    use core::future::Future;
    f$0n foo() -> impl Future<Output = usize>;
    "#,
            r#"
    use core::future::Future;
    async fn foo() -> usize;
    "#,
        );
    }

    #[test]
    fn sugar_definition_without_use() {
        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    f$0n foo() -> impl core::future::Future<Output = ()>;
    "#,
            r#"
    async fn foo();
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    f$0n foo() -> impl core::future::Future<Output = usize>;
    "#,
            r#"
    async fn foo() -> usize;
    "#,
        );
    }

    #[test]
    fn sugar_with_modifiers() {
        check_assist_not_applicable(
            toggle_async_sugar,
            r#"
    //- minicore: future
    const f$0n foo() -> impl core::future::Future<Output = ()>;
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
            //- minicore: future
            pub(crate) unsafe f$0n foo() -> impl core::future::Future<Output = usize>;
        "#,
            r#"
            pub(crate) async unsafe fn foo() -> usize;
        "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    unsafe f$0n foo() -> impl core::future::Future<Output = ()>;
    "#,
            r#"
    async unsafe fn foo();
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    unsafe extern "C" f$0n foo() -> impl core::future::Future<Output = ()>;
    "#,
            r#"
    async unsafe extern "C" fn foo();
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    f$0n foo<T>() -> impl core::future::Future<Output = T>;
    "#,
            r#"
    async fn foo<T>() -> T;
    "#,
        );

        check_assist(
            toggle_async_sugar,
            r#"
    //- minicore: future
    f$0n foo<T>() -> impl core::future::Future<Output = T>
    where
        T: Sized;
    "#,
            r#"
    async fn foo<T>() -> T
    where
        T: Sized;
    "#,
        );
    }
}
