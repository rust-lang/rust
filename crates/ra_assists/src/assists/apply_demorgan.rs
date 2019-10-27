use hir::db::HirDatabase;
use ra_syntax::ast::{self, AstNode};
use ra_syntax::SyntaxNode;

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
pub(crate) fn apply_demorgan(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let expr = ctx.find_node_at_offset::<ast::BinExpr>()?;
    let op = expr.op_kind()?;
    let op_range = expr.op_token()?.text_range();
    let opposite_op = opposite_logic_op(op)?;
    let cursor_in_range = ctx.frange.range.is_subrange(&op_range);
    if !cursor_in_range {
        return None;
    }
    let lhs = expr.lhs()?.syntax().clone();
    let lhs_range = lhs.text_range();
    let rhs = expr.rhs()?.syntax().clone();
    let rhs_range = rhs.text_range();
    let not_lhs = undo_negation(lhs)?;
    let not_rhs = undo_negation(rhs)?;

    ctx.add_action(AssistId("apply_demorgan"), "apply demorgan's law", |edit| {
        edit.target(op_range);
        edit.replace(op_range, opposite_op);
        edit.replace(lhs_range, format!("!({}", not_lhs));
        edit.replace(rhs_range, format!("{})", not_rhs));
    });
    ctx.build()
}

// Return the opposite text for a given logical operator, if it makes sense
fn opposite_logic_op(kind: ast::BinOp) -> Option<&'static str> {
    match kind {
        ast::BinOp::BooleanOr => Some("&&"),
        ast::BinOp::BooleanAnd => Some("||"),
        _ => None,
    }
}

// This function tries to undo unary negation, or inequality
fn undo_negation(node: SyntaxNode) -> Option<String> {
    match ast::Expr::cast(node)? {
        ast::Expr::BinExpr(bin) => match bin.op_kind()? {
            ast::BinOp::NegatedEqualityTest => {
                let lhs = bin.lhs()?.syntax().text();
                let rhs = bin.rhs()?.syntax().text();
                Some(format!("{} == {}", lhs, rhs))
            }
            _ => None,
        },
        ast::Expr::PrefixExpr(pe) => match pe.op_kind()? {
            ast::PrefixOp::Not => {
                let child = pe.expr()?.syntax().text();
                Some(String::from(child))
            }
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
