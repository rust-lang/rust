use ide_db::{
    EditionedFileId,
    assists::AssistId,
    defs::Definition,
    search::{FileReference, FileReferenceNode},
    syntax_helpers::node_ext::full_path_of_name_ref,
};
use syntax::{
    AstNode, SyntaxKind, TextRange,
    ast::{self, NameRef},
};

use crate::{AssistContext, Assists};

// FIXME: This ought to be a diagnostic lint.

// Assist: unnecessary_async
//
// Removes the `async` mark from functions which have no `.await` in their body.
// Looks for calls to the functions and removes the `.await` on the call site.
//
// ```
// pub asy$0nc fn foo() {}
// pub async fn bar() { foo().await }
// ```
// ->
// ```
// pub fn foo() {}
// pub async fn bar() { foo() }
// ```
pub(crate) fn unnecessary_async(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let function: ast::Fn = ctx.find_node_at_offset()?;

    // Do nothing if the cursor isn't on the async token.
    let async_token = function.async_token()?;
    if !async_token.text_range().contains_inclusive(ctx.offset()) {
        return None;
    }
    // Do nothing if the function has an `await` expression in its body.
    if function.body()?.syntax().descendants().find_map(ast::AwaitExpr::cast).is_some() {
        return None;
    }
    // Do nothing if the method is a member of trait.
    if let Some(impl_) = function.syntax().ancestors().nth(2).and_then(ast::Impl::cast)
        && impl_.trait_().is_some()
    {
        return None;
    }

    // Remove the `async` keyword plus whitespace after it, if any.
    let async_range = {
        let async_token = function.async_token()?;
        let next_token = async_token.next_token()?;
        if matches!(next_token.kind(), SyntaxKind::WHITESPACE) {
            TextRange::new(async_token.text_range().start(), next_token.text_range().end())
        } else {
            async_token.text_range()
        }
    };

    // Otherwise, we may remove the `async` keyword.
    acc.add(
        AssistId::quick_fix("unnecessary_async"),
        "Remove unnecessary async",
        async_range,
        |edit| {
            // Remove async on the function definition.
            edit.replace(async_range, "");

            // Remove all `.await`s from calls to the function we remove `async` from.
            if let Some(fn_def) = ctx.sema.to_def(&function) {
                for await_expr in find_all_references(ctx, &Definition::Function(fn_def))
                    // Keep only references that correspond NameRefs.
                    .filter_map(|(_, reference)| match reference.name {
                        FileReferenceNode::NameRef(nameref) => Some(nameref),
                        _ => None,
                    })
                    // Keep only references that correspond to await expressions
                    .filter_map(|nameref| find_await_expression(ctx, &nameref))
                {
                    if let Some(await_token) = &await_expr.await_token() {
                        edit.replace(await_token.text_range(), "");
                    }
                    if let Some(dot_token) = &await_expr.dot_token() {
                        edit.replace(dot_token.text_range(), "");
                    }
                }
            }
        },
    )
}

fn find_all_references(
    ctx: &AssistContext<'_>,
    def: &Definition,
) -> impl Iterator<Item = (EditionedFileId, FileReference)> {
    def.usages(&ctx.sema).all().into_iter().flat_map(|(file_id, references)| {
        references.into_iter().map(move |reference| (file_id, reference))
    })
}

/// Finds the await expression for the given `NameRef`.
/// If no await expression is found, returns None.
fn find_await_expression(ctx: &AssistContext<'_>, nameref: &NameRef) -> Option<ast::AwaitExpr> {
    // From the nameref, walk up the tree to the await expression.
    let await_expr = if let Some(path) = full_path_of_name_ref(nameref) {
        // Function calls.
        path.syntax()
            .parent()
            .and_then(ast::PathExpr::cast)?
            .syntax()
            .parent()
            .and_then(ast::CallExpr::cast)?
            .syntax()
            .parent()
            .and_then(ast::AwaitExpr::cast)
    } else {
        // Method calls.
        nameref
            .syntax()
            .parent()
            .and_then(ast::MethodCallExpr::cast)?
            .syntax()
            .parent()
            .and_then(ast::AwaitExpr::cast)
    };

    ctx.sema.original_ast_node(await_expr?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn applies_on_empty_function() {
        check_assist(unnecessary_async, "pub asy$0nc fn f() {}", "pub fn f() {}")
    }

    #[test]
    fn applies_and_removes_whitespace() {
        check_assist(unnecessary_async, "pub async$0       fn f() {}", "pub fn f() {}")
    }

    #[test]
    fn applies_on_function_with_a_non_await_expr() {
        check_assist(unnecessary_async, "pub asy$0nc fn f() { f2() }", "pub fn f() { f2() }")
    }

    #[test]
    fn does_not_apply_on_function_with_an_await_expr() {
        check_assist_not_applicable(unnecessary_async, "pub asy$0nc fn f() { f2().await }")
    }

    #[test]
    fn applies_and_removes_await_on_reference() {
        check_assist(
            unnecessary_async,
            r#"
pub async fn f4() { }
pub asy$0nc fn f2() { }
pub async fn f() { f2().await }
pub async fn f3() { f2().await }"#,
            r#"
pub async fn f4() { }
pub fn f2() { }
pub async fn f() { f2() }
pub async fn f3() { f2() }"#,
        )
    }

    #[test]
    fn applies_and_removes_await_from_within_module() {
        check_assist(
            unnecessary_async,
            r#"
pub async fn f4() { }
mod a { pub asy$0nc fn f2() { } }
pub async fn f() { a::f2().await }
pub async fn f3() { a::f2().await }"#,
            r#"
pub async fn f4() { }
mod a { pub fn f2() { } }
pub async fn f() { a::f2() }
pub async fn f3() { a::f2() }"#,
        )
    }

    #[test]
    fn applies_and_removes_await_on_inner_await() {
        check_assist(
            unnecessary_async,
            // Ensure that it is the first await on the 3rd line that is removed
            r#"
pub async fn f() { f2().await }
pub asy$0nc fn f2() -> i32 { 1 }
pub async fn f3() { f4(f2().await).await }
pub async fn f4(i: i32) { }"#,
            r#"
pub async fn f() { f2() }
pub fn f2() -> i32 { 1 }
pub async fn f3() { f4(f2()).await }
pub async fn f4(i: i32) { }"#,
        )
    }

    #[test]
    fn applies_and_removes_await_on_outer_await() {
        check_assist(
            unnecessary_async,
            // Ensure that it is the second await on the 3rd line that is removed
            r#"
pub async fn f() { f2().await }
pub async$0 fn f2(i: i32) { }
pub async fn f3() { f2(f4().await).await }
pub async fn f4() -> i32 { 1 }"#,
            r#"
pub async fn f() { f2() }
pub fn f2(i: i32) { }
pub async fn f3() { f2(f4().await) }
pub async fn f4() -> i32 { 1 }"#,
        )
    }

    #[test]
    fn applies_on_method_call() {
        check_assist(
            unnecessary_async,
            r#"
pub struct S { }
impl S { pub async$0 fn f2(&self) { } }
pub async fn f(s: &S) { s.f2().await }"#,
            r#"
pub struct S { }
impl S { pub fn f2(&self) { } }
pub async fn f(s: &S) { s.f2() }"#,
        )
    }

    #[test]
    fn does_not_apply_on_function_with_a_nested_await_expr() {
        check_assist_not_applicable(
            unnecessary_async,
            "async$0 fn f() { if true { loop { f2().await } } }",
        )
    }

    #[test]
    fn does_not_apply_when_not_on_async_token() {
        check_assist_not_applicable(unnecessary_async, "pub async fn$0 f() { f2() }")
    }

    #[test]
    fn does_not_apply_on_async_trait_method() {
        check_assist_not_applicable(
            unnecessary_async,
            r#"
trait Trait {
    async fn foo();
}
impl Trait for () {
    $0async fn foo() {}
}"#,
        );
    }
}
