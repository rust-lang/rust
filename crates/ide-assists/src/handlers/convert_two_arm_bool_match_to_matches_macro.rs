use syntax::ast::{self, AstNode, Pat};

use crate::{AssistContext, AssistId, AssistKind, Assists};

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
    let match_expr = ctx.find_node_at_offset::<ast::MatchExpr>()?;
    let match_arm_list = match_expr.match_arm_list()?;
    if match_arm_list.arms().count() != 2 {
        cov_mark::hit!(non_two_arm_match);
        return None;
    }

    let mut normal_arm = None;
    let mut normal_expr = None;
    let mut wildcard_expr = None;
    for arm in match_arm_list.arms() {
        if matches!(arm.pat(), Some(Pat::WildcardPat(_))) && arm.guard().is_none() {
            wildcard_expr = arm.expr();
        } else if !matches!(arm.pat(), Some(Pat::WildcardPat(_))) {
            normal_arm = Some(arm.clone());
            normal_expr = arm.expr();
        }
    }

    let invert_matches;
    if is_bool_literal_expr(&normal_expr, true) && is_bool_literal_expr(&wildcard_expr, false) {
        invert_matches = false;
    } else if is_bool_literal_expr(&normal_expr, false)
        && is_bool_literal_expr(&wildcard_expr, true)
    {
        invert_matches = true;
    } else {
        cov_mark::hit!(non_invert_bool_literal_arms);
        return None;
    }

    let target_range = ctx.sema.original_range(match_expr.syntax()).range;
    let expr = match_expr.expr()?;

    acc.add(
        AssistId("convert_two_arm_bool_match_to_matches_macro", AssistKind::RefactorRewrite),
        "Convert to matches!",
        target_range,
        |builder| {
            let mut arm_str = String::new();
            if let Some(ref pat) = normal_arm.as_ref().unwrap().pat() {
                arm_str += &pat.to_string();
            }
            if let Some(ref guard) = normal_arm.as_ref().unwrap().guard() {
                arm_str += &format!(" {}", &guard.to_string());
            }
            if invert_matches {
                builder.replace(target_range, format!("!matches!({}, {})", expr, arm_str));
            } else {
                builder.replace(target_range, format!("matches!({}, {})", expr, arm_str));
            }
        },
    )
}

fn is_bool_literal_expr(expr: &Option<ast::Expr>, expect_bool: bool) -> bool {
    if let Some(ast::Expr::Literal(lit)) = expr {
        if let ast::LiteralKind::Bool(b) = lit.kind() {
            return b == expect_bool;
        }
    }

    return false;
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
    fn not_applicable_non_bool_literal_arms() {
        cov_mark::check!(non_invert_bool_literal_arms);
        check_assist_not_applicable(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> bool {
    match a$0 {
        Some(val) => val == 3,
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
    fn not_applicable_non_bool_match() {
        cov_mark::check!(non_invert_bool_literal_arms);
        check_assist_not_applicable(
            convert_two_arm_bool_match_to_matches_macro,
            r#"
fn foo(a: Option<u32>) -> u32 {
    match a$0 {
        Some(_val) => 1,
        _ => 0
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
}
