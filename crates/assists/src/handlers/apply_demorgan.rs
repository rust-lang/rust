use syntax::ast::{self, AstNode};

use crate::{utils::invert_boolean_expression, AssistContext, AssistId, AssistKind, Assists};

// Assist: apply_demorgan
//
// Apply https://en.wikipedia.org/wiki/De_Morgan%27s_laws[De Morgan's law].
// This transforms expressions of the form `!l || !r` into `!(l && r)`.
// This also works with `&&`. This assist can only be applied with the cursor
// on either `||` or `&&`, with both operands being a negation of some kind.
// This means something of the form `!x` or `x != y`.
//
// ```
// fn main() {
//     if x != 4 ||$0 !y {}
// }
// ```
// ->
// ```
// fn main() {
//     if !(x == 4 && y) {}
// }
// ```
pub(crate) fn apply_demorgan(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let expr = ctx.find_node_at_offset::<ast::BinExpr>()?;
    let op = expr.op_kind()?;
    let op_range = expr.op_token()?.text_range();
    let opposite_op = opposite_logic_op(op)?;
    let cursor_in_range = op_range.contains_range(ctx.frange.range);
    if !cursor_in_range {
        return None;
    }

    let lhs = expr.lhs()?;
    let lhs_range = lhs.syntax().text_range();
    let not_lhs = invert_boolean_expression(lhs);

    let rhs = expr.rhs()?;
    let rhs_range = rhs.syntax().text_range();
    let not_rhs = invert_boolean_expression(rhs);

    acc.add(
        AssistId("apply_demorgan", AssistKind::RefactorRewrite),
        "Apply De Morgan's law",
        op_range,
        |edit| {
            edit.replace(op_range, opposite_op);
            edit.replace(lhs_range, format!("!({}", not_lhs.syntax().text()));
            edit.replace(rhs_range, format!("{})", not_rhs.syntax().text()));
        },
    )
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

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn demorgan_turns_and_into_or() {
        check_assist(apply_demorgan, "fn f() { !x &&$0 !x }", "fn f() { !(x || x) }")
    }

    #[test]
    fn demorgan_turns_or_into_and() {
        check_assist(apply_demorgan, "fn f() { !x ||$0 !x }", "fn f() { !(x && x) }")
    }

    #[test]
    fn demorgan_removes_inequality() {
        check_assist(apply_demorgan, "fn f() { x != x ||$0 !x }", "fn f() { !(x == x && x) }")
    }

    #[test]
    fn demorgan_general_case() {
        check_assist(apply_demorgan, "fn f() { x ||$0 x }", "fn f() { !(!x && !x) }")
    }

    #[test]
    fn demorgan_doesnt_apply_with_cursor_not_on_op() {
        check_assist_not_applicable(apply_demorgan, "fn f() { $0 !x || !x }")
    }
}
