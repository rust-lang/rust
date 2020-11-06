use hir::HirDisplay;
use syntax::{ast, AstNode, TextSize};
use test_utils::mark;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: infer_function_return_type
//
// Adds the return type to a function inferred from its tail expression if it doesn't have a return
// type specified.
//
// ```
// fn foo() { 4<|>2i32 }
// ```
// ->
// ```
// fn foo() -> i32 { 42i32 }
// ```
pub(crate) fn infer_function_return_type(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let expr = ctx.find_node_at_offset::<ast::Expr>()?;
    let func = expr.syntax().ancestors().find_map(ast::Fn::cast)?;

    if func.ret_type().is_some() {
        mark::hit!(existing_ret_type);
        return None;
    }
    let body = func.body()?;
    let tail_expr = body.expr()?;
    // check whether the expr we were at is indeed the tail expression
    if !tail_expr.syntax().text_range().contains_range(expr.syntax().text_range()) {
        mark::hit!(not_tail_expr);
        return None;
    }
    let module = ctx.sema.scope(func.syntax()).module()?;
    let ty = ctx.sema.type_of_expr(&tail_expr)?;
    let ty = ty.display_source_code(ctx.db(), module.into()).ok()?;
    let rparen = func.param_list()?.r_paren_token()?;

    acc.add(
        AssistId("change_return_type_to_result", AssistKind::RefactorRewrite),
        "Wrap return type in Result",
        tail_expr.syntax().text_range(),
        |builder| {
            let insert_pos = rparen.text_range().end() + TextSize::from(1);

            builder.insert(insert_pos, &format!("-> {} ", ty));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn infer_return_type() {
        check_assist(
            infer_function_return_type,
            r#"fn foo() {
                45<|>
            }"#,
            r#"fn foo() -> i32 {
                45
            }"#,
        );
    }

    #[test]
    fn infer_return_type_nested() {
        check_assist(
            infer_function_return_type,
            r#"fn foo() {
                if true {
                    3<|>
                } else {
                    5
                }
            }"#,
            r#"fn foo() -> i32 {
                if true {
                    3
                } else {
                    5
                }
            }"#,
        );
    }

    #[test]
    fn not_applicable_ret_type_specified() {
        mark::check!(existing_ret_type);
        check_assist_not_applicable(
            infer_function_return_type,
            r#"fn foo() -> i32 {
                ( 45<|> + 32 ) * 123
            }"#,
        );
    }

    #[test]
    fn not_applicable_non_tail_expr() {
        mark::check!(not_tail_expr);
        check_assist_not_applicable(
            infer_function_return_type,
            r#"fn foo() {
                let x = <|>3;
                ( 45 + 32 ) * 123
            }"#,
        );
    }
}
