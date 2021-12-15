use syntax::{
    ast::{edit::AstNodeEdit, make, AstNode, BlockExpr, ElseBranch, Expr, IfExpr, MatchArm},
    NodeOrToken,
    SyntaxKind::{COMMA, WHITESPACE},
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
//         Action::Move { distance } $0if distance > 10 => foo(),
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
//         Action::Move { distance } => if distance > 10 {
//             foo()
//         },
//         _ => (),
//     }
// }
// ```
pub(crate) fn move_guard_to_arm_body(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let match_arm = ctx.find_node_at_offset::<MatchArm>()?;
    let guard = match_arm.guard()?;
    if ctx.offset() > guard.syntax().text_range().end() {
        cov_mark::hit!(move_guard_unapplicable_in_arm_body);
        return None;
    }
    let space_before_guard = guard.syntax().prev_sibling_or_token();

    // FIXME: support `if let` guards too
    if guard.let_token().is_some() {
        return None;
    }
    let guard_condition = guard.expr()?;
    let arm_expr = match_arm.expr()?;
    let if_expr = make::expr_if(
        make::condition(guard_condition, None),
        make::block_expr(None, Some(arm_expr.clone())),
        None,
    )
    .indent(arm_expr.indent_level());

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
            edit.replace_ast(arm_expr, if_expr);
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
//         Action::Move { distance } => $0if distance > 10 { foo() },
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

    let mut replace_node = None;
    let if_expr: IfExpr = IfExpr::cast(arm_body.syntax().clone()).or_else(|| {
        let block_expr = BlockExpr::cast(arm_body.syntax().clone())?;
        if let Expr::IfExpr(e) = block_expr.tail_expr()? {
            replace_node = Some(block_expr.syntax().clone());
            Some(e)
        } else {
            None
        }
    })?;
    let replace_node = replace_node.unwrap_or_else(|| if_expr.syntax().clone());

    let cond = if_expr.condition()?;
    let then_block = if_expr.then_branch()?;

    // Not support moving if let to arm guard
    if cond.is_pattern_cond() {
        return None;
    }

    let buf = format!(" if {}", cond.syntax().text());

    acc.add(
        AssistId("move_arm_cond_to_match_guard", AssistKind::RefactorRewrite),
        "Move condition to match guard",
        replace_node.text_range(),
        |edit| {
            let then_only_expr = then_block.statements().next().is_none();

            match &then_block.tail_expr() {
                Some(then_expr) if then_only_expr => {
                    edit.replace(replace_node.text_range(), then_expr.syntax().text());
                    // Insert comma for expression if there isn't one
                    match match_arm.syntax().last_child_or_token() {
                        Some(NodeOrToken::Token(t)) if t.kind() == COMMA => {}
                        _ => {
                            cov_mark::hit!(move_guard_if_add_comma);
                            edit.insert(match_arm.syntax().text_range().end(), ",");
                        }
                    }
                }
                _ if replace_node != *if_expr.syntax() => {
                    // Dedent because if_expr is in a BlockExpr
                    let replace_with = then_block.dedent(1.into()).syntax().text();
                    edit.replace(replace_node.text_range(), replace_with)
                }
                _ => edit.replace(replace_node.text_range(), then_block.syntax().text()),
            }

            edit.insert(match_pat.syntax().text_range().end(), buf);

            // If with only an else branch
            if let Some(ElseBranch::Block(else_block)) = if_expr.else_branch() {
                let then_arm_end = match_arm.syntax().text_range().end();
                let else_only_expr = else_block.statements().next().is_none();
                let indent_level = match_arm.indent_level();
                let spaces = "    ".repeat(indent_level.0 as _);
                edit.insert(then_arm_end, format!("\n{}{} => ", spaces, match_pat));
                match &else_block.tail_expr() {
                    Some(else_expr) if else_only_expr => {
                        cov_mark::hit!(move_guard_ifelse_expr_only);
                        edit.insert(then_arm_end, else_expr.syntax().text());
                        edit.insert(then_arm_end, ",");
                    }
                    _ if replace_node != *if_expr.syntax() => {
                        cov_mark::hit!(move_guard_ifelse_in_block);
                        edit.insert(then_arm_end, else_block.dedent(1.into()).syntax().text());
                    }
                    _ => {
                        cov_mark::hit!(move_guard_ifelse_else_block);
                        edit.insert(then_arm_end, else_block.syntax().text());
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn move_guard_to_arm_body_range() {
        cov_mark::check!(move_guard_unapplicable_in_arm_body);
        check_assist_not_applicable(
            move_guard_to_arm_body,
            r#"
fn main() {
    match 92 {
        x if x > 10 => $0false,
        _ => true
    }
}
"#,
        );
    }
    #[test]
    fn move_guard_to_arm_body_target() {
        check_assist_target(
            move_guard_to_arm_body,
            r#"
fn main() {
    match 92 {
        x $0if x > 10 => false,
        _ => true
    }
}
"#,
            r#"if x > 10"#,
        );
    }

    #[test]
    fn move_guard_to_arm_body_works() {
        check_assist(
            move_guard_to_arm_body,
            r#"
fn main() {
    match 92 {
        x $0if x > 10 => false,
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x => if x > 10 {
            false
        },
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
fn main() {
    match 92 {
        $0x @ 4 | x @ 5    if x > 5 => true,
        _ => false
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x @ 4 | x @ 5 => if x > 5 {
            true
        },
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
fn main() {
    match 92 {
        x => if x > 10 { $0false },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
        _ => true
    }
}
"#,
        );
    }

    #[test]
    fn move_arm_cond_in_block_to_match_guard_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => {
            $0if x > 10 {
                false
            }
        },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
        _ => true
    }
}
"#,
        );
    }

    #[test]
    fn move_arm_cond_in_block_to_match_guard_add_comma_works() {
        cov_mark::check!(move_guard_if_add_comma);
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => {
            $0if x > 10 {
                false
            }
        }
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
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
fn main() {
    match 92 {
        x => if let 62 = x { $0false },
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
fn main() {
    match 92 {
        x => if x > 10 { $0 },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => {  },
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
fn main() {
    match 92 {
        x => if x > 10 {
            92;$0
            false
        },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => {
            92;
            false
        },
        _ => true
    }
}
"#,
        );
    }

    #[test]
    fn move_arm_cond_in_block_to_match_guard_if_multiline_body_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => {
            if x > 10 {
                92;$0
                false
            }
        }
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => {
            92;
            false
        }
        _ => true
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_with_else_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => if x > 10 {$0
            false
        } else {
            true
        }
        _ => true,
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
        x => true,
        _ => true,
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_with_else_block_works() {
        cov_mark::check!(move_guard_ifelse_expr_only);
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => {
            if x > 10 {$0
                false
            } else {
                true
            }
        }
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
        x => true,
        _ => true
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_else_if_empty_body_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => if x > 10 { $0 } else { },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => {  },
        x => { }
        _ => true
    }
}
"#,
        );
    }

    #[test]
    fn move_arm_cond_to_match_guard_with_else_multiline_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => if x > 10 {
            92;$0
            false
        } else {
            true
        }
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => {
            92;
            false
        }
        x => true,
        _ => true
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_with_else_multiline_else_works() {
        cov_mark::check!(move_guard_ifelse_else_block);
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => if x > 10 {$0
            false
        } else {
            42;
            true
        }
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
        x => {
            42;
            true
        }
        _ => true
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_with_else_multiline_else_block_works() {
        cov_mark::check!(move_guard_ifelse_in_block);
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => {
            if x > 10 {$0
                false
            } else {
                42;
                true
            }
        }
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
        x => {
            42;
            true
        }
        _ => true
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_with_else_last_arm_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        3 => true,
        x => {
            if x > 10 {$0
                false
            } else {
                92;
                true
            }
        }
    }
}
"#,
            r#"
fn main() {
    match 92 {
        3 => true,
        x if x > 10 => false,
        x => {
            92;
            true
        }
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_with_else_comma_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        3 => true,
        x => if x > 10 {$0
            false
        } else {
            92;
            true
        },
    }
}
"#,
            r#"
fn main() {
    match 92 {
        3 => true,
        x if x > 10 => false,
        x => {
            92;
            true
        }
    }
}
"#,
        )
    }
}
