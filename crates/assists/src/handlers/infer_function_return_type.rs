use hir::HirDisplay;
use syntax::{ast, AstNode, SyntaxToken, TextSize};
use test_utils::mark;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: infer_function_return_type
//
// Adds the return type to a function or closure inferred from its tail expression if it doesn't have a return
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
    let (tail_expr, insert_pos) = extract_tail(expr)?;
    let module = ctx.sema.scope(tail_expr.syntax()).module()?;
    let ty = ctx.sema.type_of_expr(&tail_expr).filter(|ty| !ty.is_unit())?;
    let ty = ty.display_source_code(ctx.db(), module.into()).ok()?;

    acc.add(
        AssistId("change_return_type_to_result", AssistKind::RefactorRewrite),
        "Wrap return type in Result",
        tail_expr.syntax().text_range(),
        |builder| {
            let insert_pos = insert_pos.text_range().end() + TextSize::from(1);
            builder.insert(insert_pos, &format!("-> {} ", ty));
        },
    )
}

fn extract_tail(expr: ast::Expr) -> Option<(ast::Expr, SyntaxToken)> {
    let (ret_ty, tail_expr, insert_pos) =
        if let Some(closure) = expr.syntax().ancestors().find_map(ast::ClosureExpr::cast) {
            let tail_expr = match closure.body()? {
                ast::Expr::BlockExpr(block) => block.expr()?,
                body => body,
            };
            let ret_ty = closure.ret_type();
            let rpipe = closure.param_list()?.syntax().last_token()?;
            (ret_ty, tail_expr, rpipe)
        } else {
            let func = expr.syntax().ancestors().find_map(ast::Fn::cast)?;
            let tail_expr = func.body()?.expr()?;
            let ret_ty = func.ret_type();
            let rparen = func.param_list()?.r_paren_token()?;
            (ret_ty, tail_expr, rparen)
        };
    if ret_ty.is_some() {
        mark::hit!(existing_ret_type);
        mark::hit!(existing_ret_type_closure);
        return None;
    }
    // check whether the expr we were at is indeed the tail expression
    if !tail_expr.syntax().text_range().contains_range(expr.syntax().text_range()) {
        mark::hit!(not_tail_expr);
        return None;
    }
    Some((tail_expr, insert_pos))
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

    #[test]
    fn not_applicable_unit_return_type() {
        check_assist_not_applicable(
            infer_function_return_type,
            r#"fn foo() {
                (<|>)
            }"#,
        );
    }

    #[test]
    fn infer_return_type_closure_block() {
        check_assist(
            infer_function_return_type,
            r#"fn foo() {
                |x: i32| {
                    x<|>
                };
            }"#,
            r#"fn foo() {
                |x: i32| -> i32 {
                    x
                };
            }"#,
        );
    }

    #[test]
    fn infer_return_type_closure() {
        check_assist(
            infer_function_return_type,
            r#"fn foo() {
                |x: i32| x<|>;
            }"#,
            r#"fn foo() {
                |x: i32| -> i32 x;
            }"#,
        );
    }

    #[test]
    fn infer_return_type_nested_closure() {
        check_assist(
            infer_function_return_type,
            r#"fn foo() {
                || {
                    if true {
                        3<|>
                    } else {
                        5
                    }
                }
            }"#,
            r#"fn foo() {
                || -> i32 {
                    if true {
                        3
                    } else {
                        5
                    }
                }
            }"#,
        );
    }

    #[test]
    fn not_applicable_ret_type_specified_closure() {
        mark::check!(existing_ret_type_closure);
        check_assist_not_applicable(
            infer_function_return_type,
            r#"fn foo() {
                || -> i32 { 3<|> }
            }"#,
        );
    }

    #[test]
    fn not_applicable_non_tail_expr_closure() {
        check_assist_not_applicable(
            infer_function_return_type,
            r#"fn foo() {
                || -> i32 {
                    let x = 3<|>;
                    6
                }
            }"#,
        );
    }
}
