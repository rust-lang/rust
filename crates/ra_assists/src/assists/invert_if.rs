use hir::db::HirDatabase;
use ra_syntax::ast::{self, AstNode};
use ra_syntax::T;

use crate::{Assist, AssistCtx, AssistId};

// Assist: invert_if
//
// Apply invert_if
// This transforms if expressions of the form `if !x {A} else {B}` into `if x {B} else {A}`
// This also works with `!=`. This assist can only be applied with the cursor
// on `if`.
//
// ```
// fn main() {
//     if<|> !y { A } else { B }
// }
// ```
// ->
// ```
// fn main() {
//     if y { B } else { A }
// }
// ```

pub(crate) fn invert_if(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let if_keyword = ctx.find_token_at_offset(T![if])?;
    let expr = ast::IfExpr::cast(if_keyword.parent())?;
    let if_range = if_keyword.text_range();
    let cursor_in_range = ctx.frange.range.is_subrange(&if_range);
    if !cursor_in_range {
        return None;
    }

    let cond = expr.condition()?.expr()?;
    let then_node = expr.then_branch()?.syntax().clone();

    if let ast::ElseBranch::Block(else_block) = expr.else_branch()? {
        let flip_cond = invert_boolean_expression(&cond)?;
        let cond_range = cond.syntax().text_range();
        let else_node = else_block.syntax();
        let else_range = else_node.text_range();
        let then_range = then_node.text_range();
        return ctx.add_assist(AssistId("invert_if"), "invert if branches", |edit| {
            edit.target(if_range);
            edit.replace(cond_range, flip_cond.syntax().text());
            edit.replace(else_range, then_node.text());
            edit.replace(then_range, else_node.text());
        });
    }

    None
}

pub(crate) fn invert_boolean_expression(expr: &ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::BinExpr(bin) => match bin.op_kind()? {
            ast::BinOp::NegatedEqualityTest => bin.replace_op(T![==]).map(|it| it.into()),
            _ => None,
        },
        ast::Expr::PrefixExpr(pe) => match pe.op_kind()? {
            ast::PrefixOp::Not => pe.expr(),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::helpers::{check_assist, check_assist_not_applicable};

    #[test]
    fn invert_if_remove_inequality() {
        check_assist(
            invert_if,
            "fn f() { i<|>f x != 3 { 1 } else { 3 + 2 } }",
            "fn f() { i<|>f x == 3 { 3 + 2 } else { 1 } }",
        )
    }

    #[test]
    fn invert_if_remove_not() {
        check_assist(
            invert_if,
            "fn f() { <|>if !cond { 3 * 2 } else { 1 } }",
            "fn f() { <|>if cond { 1 } else { 3 * 2 } }",
        )
    }

    #[test]
    fn invert_if_doesnt_apply_with_cursor_not_on_if() {
        check_assist_not_applicable(invert_if, "fn f() { if !<|>cond { 3 * 2 } else { 1 } }")
    }

    #[test]
    fn invert_if_doesnt_apply_without_negated() {
        check_assist_not_applicable(invert_if, "fn f() { i<|>f cond { 3 * 2 } else { 1 } }")
    }
}
