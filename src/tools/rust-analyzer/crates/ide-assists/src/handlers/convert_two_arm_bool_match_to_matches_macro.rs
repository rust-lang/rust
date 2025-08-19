use hir::Semantics;
use ide_db::RootDatabase;
use stdx::format_to;
use syntax::ast::{self, AstNode};

use crate::{AssistContext, AssistId, Assists};

// Assist: convert_two_arm_bool_match_to_matches_macro
//
// Convert 2-arm match that evaluates to a boolean into the equivalent matches! invocation.
//
// ```
// fn main() {
//     match scrutinee$0 {
//         Some(val) if val.cond() => true,
//         _ => false,
//     }
// }
// ```
// ->
// ```
// fn main() {
//     matches!(scrutinee, Some(val) if val.cond())
// }
// ```
pub(crate) fn convert_two_arm_bool_match_to_matches_macro(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    use ArmBodyExpression::*;
    let match_expr = ctx.find_node_at_offset::<ast::MatchExpr>()?;
    let match_arm_list = match_expr.match_arm_list()?;
    let mut arms = match_arm_list.arms();
    let first_arm = arms.next()?;
    let second_arm = arms.next()?;
    if arms.next().is_some() {
        cov_mark::hit!(non_two_arm_match);
        return None;
    }
    let first_arm_expr = first_arm.expr()?;
    let second_arm_expr = second_arm.expr()?;
    let first_arm_body = is_bool_literal_expr(&ctx.sema, &first_arm_expr)?;
    let second_arm_body = is_bool_literal_expr(&ctx.sema, &second_arm_expr)?;

    if !matches!(
        (&first_arm_body, &second_arm_body),
        (Literal(true), Literal(false))
            | (Literal(false), Literal(true))
            | (Expression(_), Literal(false))
    ) {
        cov_mark::hit!(non_invert_bool_literal_arms);
        return None;
    }

    let target_range = ctx.sema.original_range(match_expr.syntax()).range;
    let expr = match_expr.expr()?;

    acc.add(
        AssistId::refactor_rewrite("convert_two_arm_bool_match_to_matches_macro"),
        "Convert to matches!",
        target_range,
        |builder| {
            let mut arm_str = String::new();
            if let Some(pat) = &first_arm.pat() {
                format_to!(arm_str, "{pat}");
            }
            if let Some(guard) = &first_arm.guard() {
                arm_str += &format!(" {guard}");
            }

            let replace_with = match (first_arm_body, second_arm_body) {
                (Literal(true), Literal(false)) => {
                    format!("matches!({expr}, {arm_str})")
                }
                (Literal(false), Literal(true)) => {
                    format!("!matches!({expr}, {arm_str})")
                }
                (Expression(body_expr), Literal(false)) => {
                    arm_str.push_str(match &first_arm.guard() {
                        Some(_) => " && ",
                        _ => " if ",
                    });
                    format!("matches!({expr}, {arm_str}{body_expr})")
                }
                _ => {
                    unreachable!()
                }
            };
            builder.replace(target_range, replace_with);
        },
    )
}

enum ArmBodyExpression {
    Literal(bool),
    Expression(ast::Expr),
}

fn is_bool_literal_expr(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
) -> Option<ArmBodyExpression> {
    if let ast::Expr::Literal(lit) = expr
        && let ast::LiteralKind::Bool(b) = lit.kind()
    {
        return Some(ArmBodyExpression::Literal(b));
    }

    if !sema.type_of_expr(expr)?.original.is_bool() {
        return None;
    }

    Some(ArmBodyExpression::Expression(expr.clone()))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::convert_two_arm_bool_match_to_matches_macro;

    #[test]
    fn not_applicable_outside_of_range_left() {
        check_assist_not_applicable(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    $0 match a {
        Some(_val) => true,
        _ => false
    }
}
        "#,
        );
    }

    #[test]
    fn not_applicable_non_two_arm_match() {
        cov_mark::check!(non_two_arm_match);
        check_assist_not_applicable(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    match a$0 {
        Some(3) => true,
        Some(4) => true,
        _ => false
    }
}
        "#,
        );
    }

    #[test]
    fn not_applicable_both_false_arms() {
        cov_mark::check!(non_invert_bool_literal_arms);
        check_assist_not_applicable(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    match a$0 {
        Some(val) => false,
        _ => false
    }
}
        "#,
        );
    }

    #[test]
    fn not_applicable_both_true_arms() {
        cov_mark::check!(non_invert_bool_literal_arms);
        check_assist_not_applicable(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    match a$0 {
        Some(val) => true,
        _ => true
    }
}
        "#,
        );
    }

    #[test]
    fn convert_simple_case() {
        check_assist(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    match a$0 {
        Some(_val) => true,
        _ => false
    }
}
"#,
            r#"
fn foo(a: Option<u32>) -> bool {
    matches!(a, Some(_val))
}
"#,
        );
    }

    #[test]
    fn convert_simple_invert_case() {
        check_assist(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    match a$0 {
        Some(_val) => false,
        _ => true
    }
}
"#,
            r#"
fn foo(a: Option<u32>) -> bool {
    !matches!(a, Some(_val))
}
"#,
        );
    }

    #[test]
    fn convert_with_guard_case() {
        check_assist(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    match a$0 {
        Some(val) if val > 3 => true,
        _ => false
    }
}
"#,
            r#"
fn foo(a: Option<u32>) -> bool {
    matches!(a, Some(val) if val > 3)
}
"#,
        );
    }

    #[test]
    fn convert_enum_match_cases() {
        check_assist(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
enum X { A, B }

fn foo(a: X) -> bool {
    match a$0 {
        X::A => true,
        _ => false
    }
}
"#,
            r#"
enum X { A, B }

fn foo(a: X) -> bool {
    matches!(a, X::A)
}
"#,
        );
    }

    #[test]
    fn convert_target_simple() {
        check_assist_target(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    match a$0 {
        Some(val) => true,
        _ => false
    }
}
"#,
            r#"match a {
        Some(val) => true,
        _ => false
    }"#,
        );
    }

    #[test]
    fn convert_target_complex() {
        check_assist_target(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
enum E { X, Y }

fn main() {
    match E::X$0 {
        E::X => true,
        _ => false,
    }
}
"#,
            "match E::X {
        E::X => true,
        _ => false,
    }",
        );
    }

    #[test]
    fn convert_non_literal_bool() {
        check_assist(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn main() {
    match 0$0 {
        a @ 0..15 => a == 0,
        _ => false,
    }
}
"#,
            r#"
fn main() {
    matches!(0, a @ 0..15 if a == 0)
}
"#,
        );
        check_assist(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn main() {
    match 0$0 {
        a @ 0..15 if thing() => a == 0,
        _ => false,
    }
}
"#,
            r#"
fn main() {
    matches!(0, a @ 0..15 if thing() && a == 0)
}
"#,
        );
    }
}
