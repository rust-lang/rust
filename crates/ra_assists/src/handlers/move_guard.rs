use ra_syntax::{
    ast::{AstNode, IfExpr, MatchArm},
    SyntaxKind::WHITESPACE,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: move_guard_to_arm_body
//
// Moves match guard into match arm body.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } <|>if distance > 10 => foo(),
//         _ => (),
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } => if distance > 10 { foo() },
//         _ => (),
//     }
// }
// ```
pub(crate) fn move_guard_to_arm_body(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let match_arm = ctx.find_node_at_offset::<MatchArm>()?;
    let guard = match_arm.guard()?;
    let space_before_guard = guard.syntax().prev_sibling_or_token();

    let guard_conditions = guard.expr()?;
    let arm_expr = match_arm.expr()?;
    let buf = format!("if {} {{ {} }}", guard_conditions.syntax().text(), arm_expr.syntax().text());

    let target = guard.syntax().text_range();
    acc.add(
        AssistId("move_guard_to_arm_body", AssistKind::RefactorRewrite),
        "Move guard to arm body",
        target,
        |edit| {
            match space_before_guard {
                Some(element) if element.kind() == WHITESPACE => {
                    edit.delete(element.text_range());
                }
                _ => (),
            };

            edit.delete(guard.syntax().text_range());
            edit.replace_node_and_indent(arm_expr.syntax(), buf);
        },
    )
}

// Assist: move_arm_cond_to_match_guard
//
// Moves if expression from match arm body into a guard.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } => <|>if distance > 10 { foo() },
//         _ => (),
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } if distance > 10 => foo(),
//         _ => (),
//     }
// }
// ```
pub(crate) fn move_arm_cond_to_match_guard(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let match_arm: MatchArm = ctx.find_node_at_offset::<MatchArm>()?;
    let match_pat = match_arm.pat()?;

    let arm_body = match_arm.expr()?;
    let if_expr: IfExpr = IfExpr::cast(arm_body.syntax().clone())?;
    let cond = if_expr.condition()?;
    let then_block = if_expr.then_branch()?;

    // Not support if with else branch
    if if_expr.else_branch().is_some() {
        return None;
    }
    // Not support moving if let to arm guard
    if cond.pat().is_some() {
        return None;
    }

    let buf = format!(" if {}", cond.syntax().text());

    let target = if_expr.syntax().text_range();
    acc.add(
        AssistId("move_arm_cond_to_match_guard", AssistKind::RefactorRewrite),
        "Move condition to match guard",
        target,
        |edit| {
            let then_only_expr = then_block.statements().next().is_none();

            match &then_block.expr() {
                Some(then_expr) if then_only_expr => {
                    edit.replace(if_expr.syntax().text_range(), then_expr.syntax().text())
                }
                _ => edit.replace(if_expr.syntax().text_range(), then_block.syntax().text()),
            }

            edit.insert(match_pat.syntax().text_range().end(), buf);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn move_guard_to_arm_body_target() {
        check_assist_target(
            move_guard_to_arm_body,
            r#"
            fn f() {
                let t = 'a';
                let chars = "abcd";
                match t {
                    '\r' <|>if chars.clone().next() == Some('\n') => false,
                    _ => true
                }
            }
            "#,
            r#"if chars.clone().next() == Some('\n')"#,
        );
    }

    #[test]
    fn move_guard_to_arm_body_works() {
        check_assist(
            move_guard_to_arm_body,
            r#"
            fn f() {
                let t = 'a';
                let chars = "abcd";
                match t {
                    '\r' <|>if chars.clone().next() == Some('\n') => false,
                    _ => true
                }
            }
            "#,
            r#"
            fn f() {
                let t = 'a';
                let chars = "abcd";
                match t {
                    '\r' => if chars.clone().next() == Some('\n') { false },
                    _ => true
                }
            }
            "#,
        );
    }

    #[test]
    fn move_guard_to_arm_body_works_complex_match() {
        check_assist(
            move_guard_to_arm_body,
            r#"
            fn f() {
                match x {
                    <|>y @ 4 | y @ 5    if y > 5 => true,
                    _ => false
                }
            }
            "#,
            r#"
            fn f() {
                match x {
                    y @ 4 | y @ 5 => if y > 5 { true },
                    _ => false
                }
            }
            "#,
        );
    }

    #[test]
    fn move_arm_cond_to_match_guard_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
            fn f() {
                let t = 'a';
                let chars = "abcd";
                match t {
                    '\r' => if chars.clone().next() == Some('\n') { <|>false },
                    _ => true
                }
            }
            "#,
            r#"
            fn f() {
                let t = 'a';
                let chars = "abcd";
                match t {
                    '\r' if chars.clone().next() == Some('\n') => false,
                    _ => true
                }
            }
            "#,
        );
    }

    #[test]
    fn move_arm_cond_to_match_guard_if_let_not_works() {
        check_assist_not_applicable(
            move_arm_cond_to_match_guard,
            r#"
            fn f() {
                let t = 'a';
                let chars = "abcd";
                match t {
                    '\r' => if let Some(_) = chars.clone().next() { <|>false },
                    _ => true
                }
            }
            "#,
        );
    }

    #[test]
    fn move_arm_cond_to_match_guard_if_empty_body_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
            fn f() {
                let t = 'a';
                let chars = "abcd";
                match t {
                    '\r' => if chars.clone().next().is_some() { <|> },
                    _ => true
                }
            }
            "#,
            r#"
            fn f() {
                let t = 'a';
                let chars = "abcd";
                match t {
                    '\r' if chars.clone().next().is_some() => {  },
                    _ => true
                }
            }
            "#,
        );
    }

    #[test]
    fn move_arm_cond_to_match_guard_if_multiline_body_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
            fn f() {
                let mut t = 'a';
                let chars = "abcd";
                match t {
                    '\r' => if chars.clone().next().is_some() {
                        t = 'e';<|>
                        false
                    },
                    _ => true
                }
            }
            "#,
            r#"
            fn f() {
                let mut t = 'a';
                let chars = "abcd";
                match t {
                    '\r' if chars.clone().next().is_some() => {
                        t = 'e';
                        false
                    },
                    _ => true
                }
            }
            "#,
        );
    }
}
