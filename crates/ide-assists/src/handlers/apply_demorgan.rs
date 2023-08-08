use std::collections::VecDeque;

use syntax::{
    ast::{self, AstNode, Expr::BinExpr},
    ted::{self, Position},
    SyntaxKind,
};

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
pub(crate) fn apply_demorgan(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let mut bin_expr = ctx.find_node_at_offset::<ast::BinExpr>()?;
    let op = bin_expr.op_kind()?;
    let op_range = bin_expr.op_token()?.text_range();

    // Is the cursor on the expression's logical operator?
    if !op_range.contains_range(ctx.selection_trimmed()) {
        return None;
    }

    // Walk up the tree while we have the same binary operator
    while let Some(parent_expr) = bin_expr.syntax().parent().and_then(ast::BinExpr::cast) {
        match parent_expr.op_kind() {
            Some(parent_op) if parent_op == op => {
                bin_expr = parent_expr;
            }
            _ => break,
        }
    }

    let op = bin_expr.op_kind()?;
    let inv_token = match op {
        ast::BinaryOp::LogicOp(ast::LogicOp::And) => SyntaxKind::PIPE2,
        ast::BinaryOp::LogicOp(ast::LogicOp::Or) => SyntaxKind::AMP2,
        _ => return None,
    };

    let demorganed = bin_expr.clone_subtree().clone_for_update();

    ted::replace(demorganed.op_token()?, ast::make::token(inv_token));
    let mut exprs = VecDeque::from(vec![
        (bin_expr.lhs()?, demorganed.lhs()?),
        (bin_expr.rhs()?, demorganed.rhs()?),
    ]);

    while let Some((expr, dm)) = exprs.pop_front() {
        if let BinExpr(bin_expr) = &expr {
            if let BinExpr(cbin_expr) = &dm {
                if op == bin_expr.op_kind()? {
                    ted::replace(cbin_expr.op_token()?, ast::make::token(inv_token));
                    exprs.push_back((bin_expr.lhs()?, cbin_expr.lhs()?));
                    exprs.push_back((bin_expr.rhs()?, cbin_expr.rhs()?));
                } else {
                    let mut inv = invert_boolean_expression(expr);
                    if inv.needs_parens_in(dm.syntax().parent()?) {
                        inv = ast::make::expr_paren(inv).clone_for_update();
                    }
                    ted::replace(dm.syntax(), inv.syntax());
                }
            } else {
                return None;
            }
        } else {
            let mut inv = invert_boolean_expression(dm.clone_subtree()).clone_for_update();
            if inv.needs_parens_in(dm.syntax().parent()?) {
                inv = ast::make::expr_paren(inv).clone_for_update();
            }
            ted::replace(dm.syntax(), inv.syntax());
        }
    }

    let dm_lhs = demorganed.lhs()?;

    acc.add(
        AssistId("apply_demorgan", AssistKind::RefactorRewrite),
        "Apply De Morgan's law",
        op_range,
        |edit| {
            let paren_expr = bin_expr.syntax().parent().and_then(ast::ParenExpr::cast);
            let neg_expr = paren_expr
                .clone()
                .and_then(|paren_expr| paren_expr.syntax().parent())
                .and_then(ast::PrefixExpr::cast)
                .and_then(|prefix_expr| {
                    if prefix_expr.op_kind()? == ast::UnaryOp::Not {
                        Some(prefix_expr)
                    } else {
                        None
                    }
                });

            if let Some(paren_expr) = paren_expr {
                if let Some(neg_expr) = neg_expr {
                    cov_mark::hit!(demorgan_double_negation);
                    edit.replace_ast(ast::Expr::PrefixExpr(neg_expr), demorganed.into());
                } else {
                    cov_mark::hit!(demorgan_double_parens);
                    ted::insert_all_raw(
                        Position::before(dm_lhs.syntax()),
                        vec![
                            syntax::NodeOrToken::Token(ast::make::token(SyntaxKind::BANG)),
                            syntax::NodeOrToken::Token(ast::make::token(SyntaxKind::L_PAREN)),
                        ],
                    );

                    ted::append_child_raw(
                        demorganed.syntax(),
                        syntax::NodeOrToken::Token(ast::make::token(SyntaxKind::R_PAREN)),
                    );

                    edit.replace_ast(ast::Expr::ParenExpr(paren_expr), demorganed.into());
                }
            } else {
                ted::insert_all_raw(
                    Position::before(dm_lhs.syntax()),
                    vec![
                        syntax::NodeOrToken::Token(ast::make::token(SyntaxKind::BANG)),
                        syntax::NodeOrToken::Token(ast::make::token(SyntaxKind::L_PAREN)),
                    ],
                );
                ted::append_child_raw(demorganed.syntax(), ast::make::token(SyntaxKind::R_PAREN));
                edit.replace_ast(bin_expr, demorganed);
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

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
        check_assist(apply_demorgan, "fn f() { !(x ||$0 x) }", "fn f() { !x && !x }")
    }

    #[test]
    fn demorgan_doesnt_double_parens() {
        cov_mark::check!(demorgan_double_parens);
        check_assist(apply_demorgan, "fn f() { (x ||$0 x) }", "fn f() { !(!x && !x) }")
    }

    // FIXME : This needs to go.
    // // https://github.com/rust-lang/rust-analyzer/issues/10963
    // #[test]
    // fn demorgan_doesnt_hang() {
    //     check_assist(
    //         apply_demorgan,
    //         "fn f() { 1 || 3 &&$0 4 || 5 }",
    //         "fn f() { !(!1 || !3 || !4) || 5 }",
    //     )
    // }

    #[test]
    fn demorgan_keep_pars_for_op_precedence() {
        check_assist(
            apply_demorgan,
            "fn main() {
    let _ = !(!a ||$0 !(b || c));
}
",
            "fn main() {
    let _ = a && (b || c);
}
",
        );
    }

    #[test]
    fn demorgan_removes_pars_in_eq_precedence() {
        check_assist(
            apply_demorgan,
            "fn() { let x = a && !(!b |$0| !c); }",
            "fn() { let x = a && b && c; }",
        )
    }
}
