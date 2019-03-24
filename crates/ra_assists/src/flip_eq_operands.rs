use hir::db::HirDatabase;
use ra_syntax::{
    ast::{AstNode, BinExpr, BinOp}
};

use crate::{AssistCtx, Assist, AssistId};

pub(crate) fn flip_eq_operands(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let expr = ctx.node_at_offset::<BinExpr>()?;
    let allowed_ops = [BinOp::EqualityTest, BinOp::NegatedEqualityTest];
    let expr_op = expr.op()?;
    if ! allowed_ops.iter().any(|o| *o == expr_op) {
        return None;
    }
    let node = expr.syntax();
    let prev = node.first_child()?;
    let next = node.last_child()?;
    ctx.add_action(AssistId("flip_eq_operands"), "flip equality operands", |edit| {
        edit.target(node.range());
        edit.replace(prev.range(), next.text());
        edit.replace(next.range(), prev.text());
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
            flip_eq_operands,
            "fn f() { let res = 1 ==<|> 2; }",
            "fn f() { let res = 2 ==<|> 1; }",
        )
    }

    #[test]
    fn flip_neq_operands_for_simple_stmt() {
        check_assist(
            flip_eq_operands,
            "fn f() { let res = 1 !=<|> 2; }",
            "fn f() { let res = 2 !=<|> 1; }",
        )
    }

    #[test]
    fn flip_eq_operands_for_complex_stmt() {
        check_assist(
            flip_eq_operands,
            "fn f() { let res = (1 + 1) ==<|> (2 + 2); }",
            "fn f() { let res = (2 + 2) ==<|> (1 + 1); }",
        )
    }

    #[test]
    fn flip_eq_operands_in_match_expr() {
        check_assist(
            flip_eq_operands,
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
        check_assist_target(flip_eq_operands, "fn f() { let res = 1 ==<|> 2; }", "1 == 2")
    }
}
