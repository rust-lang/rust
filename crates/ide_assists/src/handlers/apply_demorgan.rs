use syntax::ast::{self, AstNode};
use test_utils::mark;

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
//     if !(x == 4 && !(y < 3.14)) {}
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
    let not_lhs = invert_boolean_expression(&ctx.sema, lhs);

    let rhs = expr.rhs()?;
    let rhs_range = rhs.syntax().text_range();
    let not_rhs = invert_boolean_expression(&ctx.sema, rhs);

    acc.add(
        AssistId("apply_demorgan", AssistKind::RefactorRewrite),
        "Apply De Morgan's law",
        op_range,
        |edit| {
            let paren_expr = expr.syntax().parent().and_then(|parent| ast::ParenExpr::cast(parent));

            let neg_expr = paren_expr
                .clone()
                .and_then(|paren_expr| paren_expr.syntax().parent())
                .and_then(|parent| ast::PrefixExpr::cast(parent))
                .and_then(|prefix_expr| {
                    if prefix_expr.op_kind().unwrap() == ast::PrefixOp::Not {
                        Some(prefix_expr)
                    } else {
                        None
                    }
                });

            edit.replace(op_range, opposite_op);

            if let Some(paren_expr) = paren_expr {
                edit.replace(lhs_range, not_lhs.syntax().text());
                edit.replace(rhs_range, not_rhs.syntax().text());
                if let Some(neg_expr) = neg_expr {
                    mark::hit!(demorgan_double_negation);
                    edit.replace(neg_expr.op_token().unwrap().text_range(), "");
                } else {
                    mark::hit!(demorgan_double_parens);
                    edit.replace(paren_expr.l_paren_token().unwrap().text_range(), "!(");
                }
            } else {
                edit.replace(lhs_range, format!("!({}", not_lhs.syntax().text()));
                edit.replace(rhs_range, format!("{})", not_rhs.syntax().text()));
            }
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
    use ide_db::helpers::FamousDefs;
    use test_utils::mark;

    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    const ORDABLE_FIXTURE: &'static str = r"
//- /lib.rs deps:core crate:ordable
struct NonOrderable;
struct Orderable;
impl core::cmp::Ord for Orderable {}
";

    fn check(ra_fixture_before: &str, ra_fixture_after: &str) {
        let before = &format!(
            "//- /main.rs crate:main deps:core,ordable\n{}\n{}{}",
            ra_fixture_before,
            FamousDefs::FIXTURE,
            ORDABLE_FIXTURE
        );
        check_assist(apply_demorgan, before, &format!("{}\n", ra_fixture_after));
    }

    #[test]
    fn demorgan_handles_leq() {
        check(
            r"use ordable::Orderable;
fn f() {
    Orderable < Orderable &&$0 Orderable <= Orderable
}",
            r"use ordable::Orderable;
fn f() {
    !(Orderable >= Orderable || Orderable > Orderable)
}",
        );
        check(
            r"use ordable::NonOrderable;
fn f() {
    NonOrderable < NonOrderable &&$0 NonOrderable <= NonOrderable
}",
            r"use ordable::NonOrderable;
fn f() {
    !(!(NonOrderable < NonOrderable) || !(NonOrderable <= NonOrderable))
}",
        );
    }

    #[test]
    fn demorgan_handles_geq() {
        check(
            r"use ordable::Orderable;
fn f() {
    Orderable > Orderable &&$0 Orderable >= Orderable
}",
            r"use ordable::Orderable;
fn f() {
    !(Orderable <= Orderable || Orderable < Orderable)
}",
        );
        check(
            r"use ordable::NonOrderable;
fn f() {
    Orderable > Orderable &&$0 Orderable >= Orderable
}",
            r"use ordable::NonOrderable;
fn f() {
    !(!(Orderable > Orderable) || !(Orderable >= Orderable))
}",
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
    fn demorgan_doesnt_apply_with_cursor_not_on_op() {
        check_assist_not_applicable(apply_demorgan, "fn f() { $0 !x || !x }")
    }

    #[test]
    fn demorgan_doesnt_double_negation() {
        mark::check!(demorgan_double_negation);
        check_assist(apply_demorgan, "fn f() { !(x ||$0 x) }", "fn f() { (!x && !x) }")
    }

    #[test]
    fn demorgan_doesnt_double_parens() {
        mark::check!(demorgan_double_parens);
        check_assist(apply_demorgan, "fn f() { (x ||$0 x) }", "fn f() { !(!x && !x) }")
    }
}
