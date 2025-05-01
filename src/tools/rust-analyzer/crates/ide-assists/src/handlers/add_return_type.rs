use hir::HirDisplay;
use syntax::{AstNode, SyntaxKind, SyntaxToken, TextRange, TextSize, ast, match_ast};

use crate::{AssistContext, AssistId, Assists};

// Assist: add_return_type
//
// Adds the return type to a function or closure inferred from its tail expression if it doesn't have a return
// type specified. This assists is useable in a functions or closures tail expression or return type position.
//
// ```
// fn foo() { 4$02i32 }
// ```
// ->
// ```
// fn foo() -> i32 { 42i32 }
// ```
pub(crate) fn add_return_type(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let (fn_type, tail_expr, builder_edit_pos) = extract_tail(ctx)?;
    let module = ctx.sema.scope(tail_expr.syntax())?.module();
    let ty = ctx.sema.type_of_expr(&peel_blocks(tail_expr.clone()))?.original();
    if ty.is_unit() {
        return None;
    }
    let ty = ty.display_source_code(ctx.db(), module.into(), true).ok()?;

    acc.add(
        AssistId::refactor_rewrite("add_return_type"),
        match fn_type {
            FnType::Function => "Add this function's return type",
            FnType::Closure { .. } => "Add this closure's return type",
        },
        tail_expr.syntax().text_range(),
        |builder| {
            match builder_edit_pos {
                InsertOrReplace::Insert(insert_pos, needs_whitespace) => {
                    let preceding_whitespace = if needs_whitespace { " " } else { "" };
                    builder.insert(insert_pos, format!("{preceding_whitespace}-> {ty} "))
                }
                InsertOrReplace::Replace(text_range) => {
                    builder.replace(text_range, format!("-> {ty}"))
                }
            }
            if let FnType::Closure { wrap_expr: true } = fn_type {
                cov_mark::hit!(wrap_closure_non_block_expr);
                // `|x| x` becomes `|x| -> T x` which is invalid, so wrap it in a block
                builder.replace(tail_expr.syntax().text_range(), format!("{{{tail_expr}}}"));
            }
        },
    )
}

enum InsertOrReplace {
    Insert(TextSize, bool),
    Replace(TextRange),
}

/// Check the potentially already specified return type and reject it or turn it into a builder command
/// if allowed.
fn ret_ty_to_action(
    ret_ty: Option<ast::RetType>,
    insert_after: SyntaxToken,
) -> Option<InsertOrReplace> {
    match ret_ty {
        Some(ret_ty) => match ret_ty.ty() {
            Some(ast::Type::InferType(_)) | None => {
                cov_mark::hit!(existing_infer_ret_type);
                cov_mark::hit!(existing_infer_ret_type_closure);
                Some(InsertOrReplace::Replace(ret_ty.syntax().text_range()))
            }
            _ => {
                cov_mark::hit!(existing_ret_type);
                cov_mark::hit!(existing_ret_type_closure);
                None
            }
        },
        None => {
            let insert_after_pos = insert_after.text_range().end();
            let (insert_pos, needs_whitespace) = match insert_after.next_token() {
                Some(it) if it.kind() == SyntaxKind::WHITESPACE => {
                    (insert_after_pos + TextSize::from(1), false)
                }
                _ => (insert_after_pos, true),
            };

            Some(InsertOrReplace::Insert(insert_pos, needs_whitespace))
        }
    }
}

enum FnType {
    Function,
    Closure { wrap_expr: bool },
}

/// If we're looking at a block that is supposed to return `()`, type inference
/// will just tell us it has type `()`. We have to look at the tail expression
/// to see the mismatched actual type. This 'unpeels' the various blocks to
/// hopefully let us see the type the user intends. (This still doesn't handle
/// all situations fully correctly; the 'ideal' way to handle this would be to
/// run type inference on the function again, but with a variable as the return
/// type.)
fn peel_blocks(mut expr: ast::Expr) -> ast::Expr {
    loop {
        match_ast! {
            match (expr.syntax()) {
                ast::BlockExpr(it) => {
                    if let Some(tail) = it.tail_expr() {
                        expr = tail.clone();
                    } else {
                        break;
                    }
                },
                ast::IfExpr(it) => {
                    if let Some(then_branch) = it.then_branch() {
                        expr = ast::Expr::BlockExpr(then_branch.clone());
                    } else {
                        break;
                    }
                },
                ast::MatchExpr(it) => {
                    if let Some(arm_expr) = it.match_arm_list().and_then(|l| l.arms().next()).and_then(|a| a.expr()) {
                        expr = arm_expr;
                    } else {
                        break;
                    }
                },
                _ => break,
            }
        }
    }
    expr
}

fn extract_tail(ctx: &AssistContext<'_>) -> Option<(FnType, ast::Expr, InsertOrReplace)> {
    let (fn_type, tail_expr, return_type_range, action) =
        if let Some(closure) = ctx.find_node_at_offset::<ast::ClosureExpr>() {
            let rpipe = closure.param_list()?.syntax().last_token()?;
            let rpipe_pos = rpipe.text_range().end();

            let action = ret_ty_to_action(closure.ret_type(), rpipe)?;

            let body = closure.body()?;
            let body_start = body.syntax().first_token()?.text_range().start();
            let (tail_expr, wrap_expr) = match body {
                ast::Expr::BlockExpr(block) => (block.tail_expr()?, false),
                body => (body, true),
            };

            let ret_range = TextRange::new(rpipe_pos, body_start);
            (FnType::Closure { wrap_expr }, tail_expr, ret_range, action)
        } else {
            let func = ctx.find_node_at_offset::<ast::Fn>()?;

            let rparen = func.param_list()?.r_paren_token()?;
            let rparen_pos = rparen.text_range().end();
            let action = ret_ty_to_action(func.ret_type(), rparen)?;

            let body = func.body()?;
            let stmt_list = body.stmt_list()?;
            let tail_expr = stmt_list.tail_expr()?;

            let ret_range_end = stmt_list.l_curly_token()?.text_range().start();
            let ret_range = TextRange::new(rparen_pos, ret_range_end);
            (FnType::Function, tail_expr, ret_range, action)
        };
    let range = ctx.selection_trimmed();
    if return_type_range.contains_range(range) {
        cov_mark::hit!(cursor_in_ret_position);
        cov_mark::hit!(cursor_in_ret_position_closure);
    } else if tail_expr.syntax().text_range().contains_range(range) {
        cov_mark::hit!(cursor_on_tail);
        cov_mark::hit!(cursor_on_tail_closure);
    } else {
        return None;
    }
    Some((fn_type, tail_expr, action))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn infer_return_type_specified_inferred() {
        cov_mark::check!(existing_infer_ret_type);
        check_assist(
            add_return_type,
            r#"fn foo() -> $0_ {
    45
}"#,
            r#"fn foo() -> i32 {
    45
}"#,
        );
    }

    #[test]
    fn infer_return_type_specified_inferred_closure() {
        cov_mark::check!(existing_infer_ret_type_closure);
        check_assist(
            add_return_type,
            r#"fn foo() {
    || -> _ {$045};
}"#,
            r#"fn foo() {
    || -> i32 {45};
}"#,
        );
    }

    #[test]
    fn infer_return_type_cursor_at_return_type_pos() {
        cov_mark::check!(cursor_in_ret_position);
        check_assist(
            add_return_type,
            r#"fn foo() $0{
    45
}"#,
            r#"fn foo() -> i32 {
    45
}"#,
        );
    }

    #[test]
    fn infer_return_type_cursor_at_return_type_pos_closure() {
        cov_mark::check!(cursor_in_ret_position_closure);
        check_assist(
            add_return_type,
            r#"fn foo() {
    || $045
}"#,
            r#"fn foo() {
    || -> i32 {45}
}"#,
        );
    }

    #[test]
    fn infer_return_type() {
        cov_mark::check!(cursor_on_tail);
        check_assist(
            add_return_type,
            r#"fn foo() {
    45$0
}"#,
            r#"fn foo() -> i32 {
    45
}"#,
        );
    }

    #[test]
    fn infer_return_type_no_whitespace() {
        check_assist(
            add_return_type,
            r#"fn foo(){
    45$0
}"#,
            r#"fn foo() -> i32 {
    45
}"#,
        );
    }

    #[test]
    fn infer_return_type_nested() {
        check_assist(
            add_return_type,
            r#"fn foo() {
    if true {
        3$0
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
    fn infer_return_type_nested_match() {
        check_assist(
            add_return_type,
            r#"fn foo() {
    match true {
        true => { 3$0 },
        false => { 5 },
    }
}"#,
            r#"fn foo() -> i32 {
    match true {
        true => { 3 },
        false => { 5 },
    }
}"#,
        );
    }

    #[test]
    fn not_applicable_ret_type_specified() {
        cov_mark::check!(existing_ret_type);
        check_assist_not_applicable(
            add_return_type,
            r#"fn foo() -> i32 {
    ( 45$0 + 32 ) * 123
}"#,
        );
    }

    #[test]
    fn not_applicable_non_tail_expr() {
        check_assist_not_applicable(
            add_return_type,
            r#"fn foo() {
    let x = $03;
    ( 45 + 32 ) * 123
}"#,
        );
    }

    #[test]
    fn not_applicable_unit_return_type() {
        check_assist_not_applicable(
            add_return_type,
            r#"fn foo() {
    ($0)
}"#,
        );
    }

    #[test]
    fn infer_return_type_closure_block() {
        cov_mark::check!(cursor_on_tail_closure);
        check_assist(
            add_return_type,
            r#"fn foo() {
    |x: i32| {
        x$0
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
            add_return_type,
            r#"fn foo() {
    |x: i32| { x$0 };
}"#,
            r#"fn foo() {
    |x: i32| -> i32 { x };
}"#,
        );
    }

    #[test]
    fn infer_return_type_closure_no_whitespace() {
        check_assist(
            add_return_type,
            r#"fn foo() {
    |x: i32|{ x$0 };
}"#,
            r#"fn foo() {
    |x: i32| -> i32 { x };
}"#,
        );
    }

    #[test]
    fn infer_return_type_closure_wrap() {
        cov_mark::check!(wrap_closure_non_block_expr);
        check_assist(
            add_return_type,
            r#"fn foo() {
    |x: i32| x$0;
}"#,
            r#"fn foo() {
    |x: i32| -> i32 {x};
}"#,
        );
    }

    #[test]
    fn infer_return_type_nested_closure() {
        check_assist(
            add_return_type,
            r#"fn foo() {
    || {
        if true {
            3$0
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
        cov_mark::check!(existing_ret_type_closure);
        check_assist_not_applicable(
            add_return_type,
            r#"fn foo() {
    || -> i32 { 3$0 }
}"#,
        );
    }

    #[test]
    fn not_applicable_non_tail_expr_closure() {
        check_assist_not_applicable(
            add_return_type,
            r#"fn foo() {
    || -> i32 {
        let x = 3$0;
        6
    }
}"#,
        );
    }
}
