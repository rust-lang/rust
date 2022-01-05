use std::collections::VecDeque;

use syntax::ast::{self, AstNode};

use crate::{utils::invert_boolean_expression, AssistContext, AssistId, AssistKind, Assists};

// Assist: apply_demorgan
//
// Apply https://en.wikipedia.org/wiki/De_Morgan%27s_laws[De Morgan's law].
// This transforms expressions of the form `!l || !r` into `!(l && r)`.
// This also works with `&&`. This assist can only be applied with the cursor
// on either `||` or `&&`.
//
// ```
// fn main() {
//     if x != 4 ||$0 y < 3.14 {}
// }
// ```
// ->
// ```
// fn main() {
//     if !(x == 4 && y >= 3.14) {}
// }
// ```
pub(crate) fn apply_demorgan(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let expr = ctx.find_node_at_offset::<ast::BinExpr>()?;
    let op = expr.op_kind()?;
    let op_range = expr.op_token()?.text_range();

    let opposite_op = match op {
        ast::BinaryOp::LogicOp(ast::LogicOp::And) => "||",
        ast::BinaryOp::LogicOp(ast::LogicOp::Or) => "&&",
        _ => return None,
    };

    let cursor_in_range = op_range.contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    let mut expr = expr;

    // Walk up the tree while we have the same binary operator
    while let Some(parent_expr) = expr.syntax().parent().and_then(ast::BinExpr::cast) {
        match expr.op_kind() {
            Some(parent_op) if parent_op == op => {
                expr = parent_expr;
            }
            _ => break,
        }
    }

    let mut expr_stack = vec![expr.clone()];
    let mut terms = Vec::new();
    let mut op_ranges = Vec::new();

    // Find all the children with the same binary operator
    while let Some(expr) = expr_stack.pop() {
        let mut traverse_bin_expr_arm = |expr| {
            if let ast::Expr::BinExpr(bin_expr) = expr {
                if let Some(expr_op) = bin_expr.op_kind() {
                    if expr_op == op {
                        expr_stack.push(bin_expr);
                    } else {
                        terms.push(ast::Expr::BinExpr(bin_expr));
                    }
                } else {
                    terms.push(ast::Expr::BinExpr(bin_expr));
                }
            } else {
                terms.push(expr);
            }
        };

        op_ranges.extend(expr.op_token().map(|t| t.text_range()));
        traverse_bin_expr_arm(expr.lhs()?);
        traverse_bin_expr_arm(expr.rhs()?);
    }

    acc.add(
        AssistId("apply_demorgan", AssistKind::RefactorRewrite),
        "Apply De Morgan's law",
        op_range,
        |edit| {
            terms.sort_by_key(|t| t.syntax().text_range().start());
            let mut terms = VecDeque::from(terms);

            let paren_expr = expr.syntax().parent().and_then(ast::ParenExpr::cast);

            let neg_expr = paren_expr
                .clone()
                .and_then(|paren_expr| paren_expr.syntax().parent())
                .and_then(ast::PrefixExpr::cast)
                .and_then(|prefix_expr| {
                    if prefix_expr.op_kind().unwrap() == ast::UnaryOp::Not {
                        Some(prefix_expr)
                    } else {
                        None
                    }
                });

            for op_range in op_ranges {
                edit.replace(op_range, opposite_op);
            }

            if let Some(paren_expr) = paren_expr {
                for term in terms {
                    let range = term.syntax().text_range();
                    let not_term = invert_boolean_expression(term);

                    edit.replace(range, not_term.syntax().text());
                }

                if let Some(neg_expr) = neg_expr {
                    cov_mark::hit!(demorgan_double_negation);
                    edit.replace(neg_expr.op_token().unwrap().text_range(), "");
                } else {
                    cov_mark::hit!(demorgan_double_parens);
                    edit.replace(paren_expr.l_paren_token().unwrap().text_range(), "!(");
                }
            } else {
                if let Some(lhs) = terms.pop_front() {
                    let lhs_range = lhs.syntax().text_range();
                    let not_lhs = invert_boolean_expression(lhs);

                    edit.replace(lhs_range, format!("!({}", not_lhs.syntax().text()));
                }

                if let Some(rhs) = terms.pop_back() {
                    let rhs_range = rhs.syntax().text_range();
                    let not_rhs = invert_boolean_expression(rhs);

                    edit.replace(rhs_range, format!("{})", not_rhs.syntax().text()));
                }

                for term in terms {
                    let term_range = term.syntax().text_range();
                    let not_term = invert_boolean_expression(term);
                    edit.replace(term_range, not_term.syntax().text());
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn demorgan_handles_leq() {
        check_assist(
            apply_demorgan,
            r#"
struct S;
fn f() { S < S &&$0 S <= S }
"#,
            r#"
struct S;
fn f() { !(S >= S || S > S) }
"#,
        );
    }

    #[test]
    fn demorgan_handles_geq() {
        check_assist(
            apply_demorgan,
            r#"
struct S;
fn f() { S > S &&$0 S >= S }
"#,
            r#"
struct S;
fn f() { !(S <= S || S < S) }
"#,
        );
    }

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
    fn demorgan_multiple_terms() {
        check_assist(apply_demorgan, "fn f() { x ||$0 y || z }", "fn f() { !(!x && !y && !z) }");
        check_assist(apply_demorgan, "fn f() { x || y ||$0 z }", "fn f() { !(!x && !y && !z) }");
    }

    #[test]
    fn demorgan_doesnt_apply_with_cursor_not_on_op() {
        check_assist_not_applicable(apply_demorgan, "fn f() { $0 !x || !x }")
    }

    #[test]
    fn demorgan_doesnt_double_negation() {
        cov_mark::check!(demorgan_double_negation);
        check_assist(apply_demorgan, "fn f() { !(x ||$0 x) }", "fn f() { (!x && !x) }")
    }

    #[test]
    fn demorgan_doesnt_double_parens() {
        cov_mark::check!(demorgan_double_parens);
        check_assist(apply_demorgan, "fn f() { (x ||$0 x) }", "fn f() { !(!x && !x) }")
    }

    // https://github.com/rust-analyzer/rust-analyzer/issues/10963
    #[test]
    fn demorgan_doesnt_hang() {
        check_assist(
            apply_demorgan,
            "fn f() { 1 || 3 &&$0 4 || 5 }",
            "fn f() { !(!1 || !3 || !4) || 5 }",
        )
    }
}
