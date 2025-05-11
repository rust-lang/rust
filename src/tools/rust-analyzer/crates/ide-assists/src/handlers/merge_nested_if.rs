use ide_db::syntax_helpers::node_ext::is_pattern_cond;
use syntax::{
    T,
    ast::{self, AstNode, BinaryOp},
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
};
// Assist: merge_nested_if
//
// This transforms if expressions of the form `if x { if y {A} }` into `if x && y {A}`
// This assist can only be applied with the cursor on `if`.
//
// ```
// fn main() {
//    i$0f x == 3 { if y == 4 { 1 } }
// }
// ```
// ->
// ```
// fn main() {
//    if x == 3 && y == 4 { 1 }
// }
// ```
pub(crate) fn merge_nested_if(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let if_keyword = ctx.find_token_syntax_at_offset(T![if])?;
    let expr = ast::IfExpr::cast(if_keyword.parent()?)?;
    let if_range = if_keyword.text_range();
    let cursor_in_range = if_range.contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    //should not apply to if with else branch.
    if expr.else_branch().is_some() {
        return None;
    }

    let cond = expr.condition()?;
    //should not apply for if-let
    if is_pattern_cond(cond.clone()) {
        return None;
    }

    let cond_range = cond.syntax().text_range();

    //check if the then branch is a nested if
    let then_branch = expr.then_branch()?;
    let stmt = then_branch.stmt_list()?;
    if stmt.statements().count() != 0 {
        return None;
    }

    let nested_if_to_merge = then_branch.tail_expr().and_then(|e| match e {
        ast::Expr::IfExpr(e) => Some(e),
        _ => None,
    })?;
    // should not apply to nested if with else branch.
    if nested_if_to_merge.else_branch().is_some() {
        return None;
    }
    let nested_if_cond = nested_if_to_merge.condition()?;
    if is_pattern_cond(nested_if_cond.clone()) {
        return None;
    }

    let nested_if_then_branch = nested_if_to_merge.then_branch()?;
    let then_branch_range = then_branch.syntax().text_range();

    acc.add(AssistId::refactor_rewrite("merge_nested_if"), "Merge nested if", if_range, |edit| {
        let cond_text = if has_logic_op_or(&cond) {
            format!("({})", cond.syntax().text())
        } else {
            cond.syntax().text().to_string()
        };

        let nested_if_cond_text = if has_logic_op_or(&nested_if_cond) {
            format!("({})", nested_if_cond.syntax().text())
        } else {
            nested_if_cond.syntax().text().to_string()
        };

        let replace_cond = format!("{cond_text} && {nested_if_cond_text}");

        edit.replace(cond_range, replace_cond);
        edit.replace(then_branch_range, nested_if_then_branch.syntax().text());
    })
}

/// Returns whether the given if condition has logical operators.
fn has_logic_op_or(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::BinExpr(bin_expr) => {
            if let Some(kind) = bin_expr.op_kind() {
                matches!(kind, BinaryOp::LogicOp(ast::LogicOp::Or))
            } else {
                false
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn merge_nested_if_test1() {
        check_assist(
            merge_nested_if,
            "fn f() { i$0f x == 3 { if y == 4 { 1 } } }",
            "fn f() { if x == 3 && y == 4 { 1 } }",
        )
    }

    #[test]
    fn merge_nested_if_test2() {
        check_assist(
            merge_nested_if,
            "fn f() { i$0f x == 3 || y == 1 { if z == 4 { 1 } } }",
            "fn f() { if (x == 3 || y == 1) && z == 4 { 1 } }",
        )
    }

    #[test]
    fn merge_nested_if_test3() {
        check_assist(
            merge_nested_if,
            "fn f() { i$0f x == 3 && y == 1 { if z == 4 { 1 } } }",
            "fn f() { if x == 3 && y == 1 && z == 4 { 1 } }",
        )
    }

    #[test]
    fn merge_nested_if_test4() {
        check_assist(
            merge_nested_if,
            "fn f() { i$0f x == 3 && y == 1 { if z == 4 && q == 3 { 1 } } }",
            "fn f() { if x == 3 && y == 1 && z == 4 && q == 3 { 1 } }",
        )
    }

    #[test]
    fn merge_nested_if_test5() {
        check_assist(
            merge_nested_if,
            "fn f() { i$0f x == 3 && y == 1 { if z == 4 || q == 3 { 1 } } }",
            "fn f() { if x == 3 && y == 1 && (z == 4 || q == 3) { 1 } }",
        )
    }

    #[test]
    fn merge_nested_if_test6() {
        check_assist(
            merge_nested_if,
            "fn f() { i$0f x == 3 || y == 1 { if z == 4 || q == 3 { 1 } } }",
            "fn f() { if (x == 3 || y == 1) && (z == 4 || q == 3) { 1 } }",
        )
    }

    #[test]
    fn merge_nested_if_test7() {
        check_assist(
            merge_nested_if,
            "fn f() { i$0f x == 3 || y == 1 { if z == 4 && q == 3 { 1 } } }",
            "fn f() { if (x == 3 || y == 1) && z == 4 && q == 3 { 1 } }",
        )
    }

    #[test]
    fn merge_nested_if_do_not_apply_to_if_with_else_branch() {
        check_assist_not_applicable(
            merge_nested_if,
            "fn f() { i$0f x == 3 { if y == 4 { 1 } } else { 2 } }",
        )
    }

    #[test]
    fn merge_nested_if_do_not_apply_to_nested_if_with_else_branch() {
        check_assist_not_applicable(
            merge_nested_if,
            "fn f() { i$0f x == 3 { if y == 4 { 1 } else { 2 } } }",
        )
    }

    #[test]
    fn merge_nested_if_do_not_apply_to_if_let() {
        check_assist_not_applicable(
            merge_nested_if,
            "fn f() { i$0f let Some(x) = y { if x == 4 { 1 } } }",
        )
    }

    #[test]
    fn merge_nested_if_do_not_apply_to_nested_if_let() {
        check_assist_not_applicable(
            merge_nested_if,
            "fn f() { i$0f y == 0 { if let Some(x) = y { 1 } } }",
        )
    }

    #[test]
    fn merge_nested_if_do_not_apply_to_if_with_else_branch_and_nested_if() {
        check_assist_not_applicable(
            merge_nested_if,
            "fn f() { i$0f x == 3 { if y == 4 { 1 } } else { if z == 5 { 2 } } }",
        )
    }

    #[test]
    fn merge_nested_if_do_not_apply_with_cursor_not_on_if() {
        check_assist_not_applicable(merge_nested_if, "fn f() { if $0x==0 { if y == 3 { 1 } } }")
    }

    #[test]
    fn merge_nested_if_do_not_apply_with_mulpiple_if() {
        check_assist_not_applicable(
            merge_nested_if,
            "fn f() { i$0f x == 0 { if y == 3 { 1 } else if y == 4 { 2 } } }",
        )
    }
    #[test]
    fn merge_nested_if_do_not_apply_with_not_only_has_nested_if() {
        check_assist_not_applicable(
            merge_nested_if,
            "fn f() { i$0f x == 0 { if y == 3 { foo(); } foo(); } }",
        )
    }

    #[test]
    fn merge_nested_if_do_not_apply_with_multiply_nested_if() {
        check_assist_not_applicable(
            merge_nested_if,
            "fn f() { i$0f x == 0 { if y == 3 { foo(); } if z == 3 { 2 } } }",
        )
    }
}
