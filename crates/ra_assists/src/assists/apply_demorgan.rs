use super::invert_if::invert_boolean_expression;
use hir::db::HirDatabase;
use ra_syntax::ast::{self, AstNode};

use crate::{Assist, AssistCtx, AssistId};

// Assist: apply_demorgan
//
// Apply [De Morgan's law](https://en.wikipedia.org/wiki/De_Morgan%27s_laws).
// This transforms expressions of the form `!l || !r` into `!(l && r)`.
// This also works with `&&`. This assist can only be applied with the cursor
// on either `||` or `&&`, with both operands being a negation of some kind.
// This means something of the form `!x` or `x != y`.
//
// ```
// fn main() {
//     if x != 4 ||<|> !y {}
// }
// ```
// ->
// ```
// fn main() {
//     if !(x == 4 && y) {}
// }
// ```
pub(crate) fn apply_demorgan(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let expr = ctx.find_node_at_offset::<ast::BinExpr>()?;
    let op = expr.op_kind()?;
    let op_range = expr.op_token()?.text_range();
    let opposite_op = opposite_logic_op(op)?;
    let cursor_in_range = ctx.frange.range.is_subrange(&op_range);
    if !cursor_in_range {
        return None;
    }
    let lhs = expr.lhs()?;
    let lhs_range = lhs.syntax().text_range();
    let rhs = expr.rhs()?;
    let rhs_range = rhs.syntax().text_range();
    let not_lhs = invert_boolean_expression(&lhs)?;
    let not_rhs = invert_boolean_expression(&rhs)?;

    ctx.add_assist(AssistId("apply_demorgan"), "apply demorgan's law", |edit| {
        edit.target(op_range);
        edit.replace(op_range, opposite_op);
        edit.replace(lhs_range, format!("!({}", not_lhs.syntax().text()));
        edit.replace(rhs_range, format!("{})", not_rhs.syntax().text()));
    })
}

// Return the opposite text for a given logical operator, if it makes sense
fn opposite_logic_op(kind: ast::BinOp) -> Option<&'static str> {
    match kind {
        ast::BinOp::BooleanOr => Some("&&"),
        ast::BinOp::BooleanAnd => Some("||"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::helpers::{check_assist, check_assist_not_applicable};

    #[test]
    fn demorgan_turns_and_into_or() {
        check_assist(apply_demorgan, "fn f() { !x &&<|> !x }", "fn f() { !(x ||<|> x) }")
    }

    #[test]
    fn demorgan_turns_or_into_and() {
        check_assist(apply_demorgan, "fn f() { !x ||<|> !x }", "fn f() { !(x &&<|> x) }")
    }

    #[test]
    fn demorgan_removes_inequality() {
        check_assist(apply_demorgan, "fn f() { x != x ||<|> !x }", "fn f() { !(x == x &&<|> x) }")
    }

    #[test]
    fn demorgan_doesnt_apply_with_cursor_not_on_op() {
        check_assist_not_applicable(apply_demorgan, "fn f() { <|> !x || !x }")
    }

    #[test]
    fn demorgan_doesnt_apply_when_operands_arent_negated_already() {
        check_assist_not_applicable(apply_demorgan, "fn f() { x ||<|> x }")
    }
}
