use syntax::{
    SyntaxKind, T,
    ast::{self, AstNode, BinExpr, syntax_factory::SyntaxFactory},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: flip_binexpr
//
// Flips operands of a binary expression.
//
// ```
// fn main() {
//     let _ = 90 +$0 2;
// }
// ```
// ->
// ```
// fn main() {
//     let _ = 2 + 90;
// }
// ```
pub(crate) fn flip_binexpr(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let expr = ctx.find_node_at_offset::<BinExpr>()?;
    let lhs = expr.lhs()?;
    let rhs = expr.rhs()?;

    let lhs = match &lhs {
        ast::Expr::BinExpr(bin_expr) if bin_expr.op_kind() == expr.op_kind() => bin_expr.rhs()?,
        _ => lhs,
    };

    let op_token = expr.op_token()?;
    // The assist should be applied only if the cursor is on the operator
    let cursor_in_range = op_token.text_range().contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }
    let action: FlipAction = expr.op_kind()?.into();
    // The assist should not be applied for certain operators
    if let FlipAction::DontFlip = action {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("flip_binexpr"),
        "Flip binary expression",
        op_token.text_range(),
        |builder| {
            let mut editor = builder.make_editor(&expr.syntax().parent().unwrap());
            let make = SyntaxFactory::with_mappings();
            if let FlipAction::FlipAndReplaceOp(binary_op) = action {
                editor.replace(op_token, make.token(binary_op))
            };
            editor.replace(lhs.syntax(), rhs.syntax());
            editor.replace(rhs.syntax(), lhs.syntax());
            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

enum FlipAction {
    // Flip the expression
    Flip,
    // Flip the expression and replace the operator with this string
    FlipAndReplaceOp(SyntaxKind),
    // Do not flip the expression
    DontFlip,
}

impl From<ast::BinaryOp> for FlipAction {
    fn from(op_kind: ast::BinaryOp) -> Self {
        match op_kind {
            ast::BinaryOp::Assignment { .. } => FlipAction::DontFlip,
            ast::BinaryOp::CmpOp(ast::CmpOp::Ord { ordering, strict }) => {
                let rev_op = match (ordering, strict) {
                    (ast::Ordering::Less, true) => T![>],
                    (ast::Ordering::Less, false) => T![>=],
                    (ast::Ordering::Greater, true) => T![<],
                    (ast::Ordering::Greater, false) => T![<=],
                };
                FlipAction::FlipAndReplaceOp(rev_op)
            }
            _ => FlipAction::Flip,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn flip_binexpr_target_is_the_op() {
        check_assist_target(flip_binexpr, "fn f() { let res = 1 ==$0 2; }", "==")
    }

    #[test]
    fn flip_binexpr_not_applicable_for_assignment() {
        check_assist_not_applicable(flip_binexpr, "fn f() { let mut _x = 1; _x +=$0 2 }")
    }

    #[test]
    fn flip_binexpr_works_for_eq() {
        check_assist(flip_binexpr, "fn f() { let res = 1 ==$0 2; }", "fn f() { let res = 2 == 1; }")
    }

    #[test]
    fn flip_binexpr_works_for_gt() {
        check_assist(flip_binexpr, "fn f() { let res = 1 >$0 2; }", "fn f() { let res = 2 < 1; }")
    }

    #[test]
    fn flip_binexpr_works_for_lteq() {
        check_assist(flip_binexpr, "fn f() { let res = 1 <=$0 2; }", "fn f() { let res = 2 >= 1; }")
    }

    #[test]
    fn flip_binexpr_works_for_complex_expr() {
        check_assist(
            flip_binexpr,
            "fn f() { let res = (1 + 1) ==$0 (2 + 2); }",
            "fn f() { let res = (2 + 2) == (1 + 1); }",
        )
    }

    #[test]
    fn flip_binexpr_works_for_lhs_arith() {
        check_assist(
            flip_binexpr,
            r"fn f() { let res = 1 + (2 - 3) +$0 4 + 5; }",
            r"fn f() { let res = 1 + 4 + (2 - 3) + 5; }",
        )
    }

    #[test]
    fn flip_binexpr_works_for_lhs_cmp() {
        check_assist(
            flip_binexpr,
            r"fn f() { let res = 1 + (2 - 3) >$0 4 + 5; }",
            r"fn f() { let res = 4 + 5 < 1 + (2 - 3); }",
        )
    }

    #[test]
    fn flip_binexpr_works_inside_match() {
        check_assist(
            flip_binexpr,
            r#"
            fn dyn_eq(&self, other: &dyn Diagnostic) -> bool {
                match other.downcast_ref::<Self>() {
                    None => false,
                    Some(it) => it ==$0 self,
                }
            }
            "#,
            r#"
            fn dyn_eq(&self, other: &dyn Diagnostic) -> bool {
                match other.downcast_ref::<Self>() {
                    None => false,
                    Some(it) => self == it,
                }
            }
            "#,
        )
    }
}
