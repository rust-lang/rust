use hir::db::HirDatabase;
use ra_syntax::ast::{AstNode, BinExpr, BinOp};

use crate::{AssistCtx, Assist, AssistId};

/// Flip binary comparison expressions (==, !=, >, >=, <, <=).
pub(crate) fn flip_binexpr(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let expr = ctx.node_at_offset::<BinExpr>()?;
    let lhs = expr.lhs()?.syntax();
    let rhs = expr.rhs()?.syntax();
    let op_range = expr.op()?.range();
    // The assist should be available only if the cursor is on the operator
    let cursor_in_range = ctx.frange.range.is_subrange(&op_range);
    // The assist should be available only for these binary operators
    // (it should not change the meaning of the expression)
    let allowed_ops = [
        BinOp::EqualityTest,
        BinOp::NegatedEqualityTest,
        BinOp::GreaterTest,
        BinOp::GreaterEqualTest,
        BinOp::LesserTest,
        BinOp::LesserEqualTest,
    ];
    let op_kind = expr.op_kind()?;
    if !cursor_in_range || !allowed_ops.iter().any(|o| *o == op_kind) {
        return None;
    }
    let new_op = match op_kind {
        BinOp::GreaterTest => Some("<"),
        BinOp::GreaterEqualTest => Some("<="),
        BinOp::LesserTest => Some(">"),
        BinOp::LesserEqualTest => Some(">="),
        _ => None,
    };
    ctx.add_action(AssistId("flip_binexpr"), "flip binary expression", |edit| {
        edit.target(op_range);
        if let Some(new_op) = new_op {
            edit.replace(op_range, new_op);
        }
        edit.replace(lhs.range(), rhs.text());
        edit.replace(rhs.range(), lhs.text());
    });

    ctx.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::helpers::{check_assist, check_assist_target};

    #[test]
    fn flip_eq_operands_for_simple_stmt() {
        check_assist(
            flip_binexpr,
            "fn f() { let res = 1 ==<|> 2; }",
            "fn f() { let res = 2 ==<|> 1; }",
        )
    }

    #[test]
    fn flip_neq_operands_for_simple_stmt() {
        check_assist(
            flip_binexpr,
            "fn f() { let res = 1 !=<|> 2; }",
            "fn f() { let res = 2 !=<|> 1; }",
        )
    }

    #[test]
    fn flip_gt_operands_for_simple_stmt() {
        check_assist(
            flip_binexpr,
            "fn f() { let res = 1 ><|> 2; }",
            "fn f() { let res = 2 <<|> 1; }",
        )
    }

    #[test]
    fn flip_gteq_operands_for_simple_stmt() {
        check_assist(
            flip_binexpr,
            "fn f() { let res = 1 >=<|> 2; }",
            "fn f() { let res = 2 <=<|> 1; }",
        )
    }

    #[test]
    fn flip_lt_operands_for_simple_stmt() {
        check_assist(
            flip_binexpr,
            "fn f() { let res = 1 <<|> 2; }",
            "fn f() { let res = 2 ><|> 1; }",
        )
    }

    #[test]
    fn flip_lteq_operands_for_simple_stmt() {
        check_assist(
            flip_binexpr,
            "fn f() { let res = 1 <=<|> 2; }",
            "fn f() { let res = 2 >=<|> 1; }",
        )
    }

    #[test]
    fn flip_eq_operands_for_complex_stmt() {
        check_assist(
            flip_binexpr,
            "fn f() { let res = (1 + 1) ==<|> (2 + 2); }",
            "fn f() { let res = (2 + 2) ==<|> (1 + 1); }",
        )
    }

    #[test]
    fn flip_eq_operands_in_match_expr() {
        check_assist(
            flip_binexpr,
            r#"
            fn dyn_eq(&self, other: &dyn Diagnostic) -> bool {
                match other.downcast_ref::<Self>() {
                    None => false,
                    Some(it) => it ==<|> self,
                }
            }
            "#,
            r#"
            fn dyn_eq(&self, other: &dyn Diagnostic) -> bool {
                match other.downcast_ref::<Self>() {
                    None => false,
                    Some(it) => self ==<|> it,
                }
            }
            "#,
        )
    }

    #[test]
    fn flip_eq_operands_target() {
        check_assist_target(flip_binexpr, "fn f() { let res = 1 ==<|> 2; }", "==")
    }

    #[test]
    fn flip_gt_operands_target() {
        check_assist_target(flip_binexpr, "fn f() { let res = 1 ><|> 2; }", ">")
    }

}
