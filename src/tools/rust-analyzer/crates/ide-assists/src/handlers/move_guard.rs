use itertools::Itertools;
use syntax::{
    SyntaxKind::WHITESPACE,
    ast::{
        AstNode, BlockExpr, ElseBranch, Expr, IfExpr, MatchArm, Pat, edit::AstNodeEdit, make,
        prec::ExprPrecedence, syntax_factory::SyntaxFactory,
    },
    syntax_editor::Element,
};

use crate::{AssistContext, AssistId, Assists};

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
pub(crate) fn move_guard_to_arm_body(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let match_arm = ctx.find_node_at_offset::<MatchArm>()?;
    let guard = match_arm.guard()?;
    if ctx.offset() > guard.syntax().text_range().end() {
        cov_mark::hit!(move_guard_inapplicable_in_arm_body);
        return None;
    }
    let space_before_guard = guard.syntax().prev_sibling_or_token();
    let space_after_arrow = match_arm.fat_arrow_token()?.next_sibling_or_token();

    let guard_condition = guard.condition()?.reset_indent();
    let arm_expr = match_arm.expr()?;
    let then_branch = crate::utils::wrap_block(&arm_expr);
    let if_expr = make::expr_if(guard_condition, then_branch, None).indent(arm_expr.indent_level());

    let target = guard.syntax().text_range();
    acc.add(
        AssistId::refactor_rewrite("move_guard_to_arm_body"),
        "Move guard to arm body",
        target,
        |builder| {
            let mut edit = builder.make_editor(match_arm.syntax());
            if let Some(element) = space_before_guard
                && element.kind() == WHITESPACE
            {
                edit.delete(element);
            }
            if let Some(element) = space_after_arrow
                && element.kind() == WHITESPACE
            {
                edit.replace(element, make::tokens::single_space());
            }

            edit.delete(guard.syntax());
            edit.replace(arm_expr.syntax(), if_expr.syntax());
            builder.add_file_edits(ctx.vfs_file_id(), edit);
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
pub(crate) fn move_arm_cond_to_match_guard(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let match_arm: MatchArm = ctx.find_node_at_offset::<MatchArm>()?;
    let match_pat = match_arm.pat()?;
    let arm_body = match_arm.expr()?;
    let arm_guard = match_arm.guard().and_then(|it| it.condition());

    let mut replace_node = None;
    let if_expr: IfExpr = IfExpr::cast(arm_body.syntax().clone()).or_else(|| {
        let block_expr = BlockExpr::cast(arm_body.syntax().clone())?;
        if block_expr.statements().next().is_some() {
            cov_mark::hit!(move_guard_non_naked_if);
            return None;
        }
        if let Expr::IfExpr(e) = block_expr.tail_expr()? {
            replace_node = Some(block_expr.syntax().clone());
            Some(e)
        } else {
            None
        }
    })?;
    if ctx.offset() > if_expr.then_branch()?.syntax().text_range().start() {
        return None;
    }

    let replace_node = replace_node.unwrap_or_else(|| if_expr.syntax().clone());
    let needs_dedent = replace_node != *if_expr.syntax();
    let (conds_blocks, tail) = parse_if_chain(if_expr)?;

    acc.add(
        AssistId::refactor_rewrite("move_arm_cond_to_match_guard"),
        "Move condition to match guard",
        replace_node.text_range(),
        |builder| {
            let make = SyntaxFactory::without_mappings();
            let mut replace_arms = vec![];

            // Dedent if if_expr is in a BlockExpr
            let dedent = if needs_dedent {
                cov_mark::hit!(move_guard_ifelse_in_block);
                1
            } else {
                cov_mark::hit!(move_guard_ifelse_else_block);
                0
            };
            let indent_level = match_arm.indent_level();
            let make_guard = |cond: Option<Expr>| {
                let condition = match (arm_guard.clone(), cond) {
                    (None, None) => return None,
                    (None, Some(it)) | (Some(it), None) => it,
                    (Some(lhs), Some(rhs)) => {
                        let op_expr = |expr: Expr| {
                            if expr.precedence().needs_parentheses_in(ExprPrecedence::LAnd) {
                                make.expr_paren(expr).into()
                            } else {
                                expr
                            }
                        };
                        let op = syntax::ast::BinaryOp::LogicOp(syntax::ast::LogicOp::And);
                        let expr_bin = make.expr_bin(op_expr(lhs), op, op_expr(rhs));
                        expr_bin.into()
                    }
                };
                Some(make.match_guard(condition))
            };

            for (cond, block) in conds_blocks {
                let only_expr = block.statements().next().is_none();
                let expr = match block.tail_expr() {
                    Some(then_expr) if only_expr => then_expr,
                    _ => block.dedent(dedent.into()).into(),
                };
                let new_arm = make.match_arm(match_pat.clone(), make_guard(Some(cond)), expr);
                replace_arms.push(new_arm);
            }
            if let Some(block) = tail {
                cov_mark::hit!(move_guard_ifelse_else_tail);
                let only_expr = block.statements().next().is_none();
                let expr = match block.tail_expr() {
                    Some(expr) if only_expr => {
                        cov_mark::hit!(move_guard_ifelse_expr_only);
                        expr
                    }
                    _ => block.dedent(dedent.into()).into(),
                };
                let new_arm = make.match_arm(match_pat, make_guard(None), expr);
                replace_arms.push(new_arm);
            } else {
                // There's no else branch. Add a pattern without guard, unless the following match
                // arm is `_ => ...`
                cov_mark::hit!(move_guard_ifelse_notail);
                match match_arm.syntax().next_sibling().and_then(MatchArm::cast) {
                    Some(next_arm)
                        if matches!(next_arm.pat(), Some(Pat::WildcardPat(_)))
                            && next_arm.guard().is_none() =>
                    {
                        cov_mark::hit!(move_guard_ifelse_has_wildcard);
                    }
                    _ => {
                        let block_expr = make.expr_empty_block().into();
                        replace_arms.push(make.match_arm(match_pat, make_guard(None), block_expr));
                    }
                }
            }

            let mut edit = builder.make_editor(match_arm.syntax());

            let newline = make.whitespace(&format!("\n{indent_level}"));
            let replace_arms = replace_arms.iter().map(|it| it.syntax().syntax_element());
            let replace_arms = Itertools::intersperse(replace_arms, newline.syntax_element());
            edit.replace_with_many(match_arm.syntax(), replace_arms.collect());

            builder.add_file_edits(ctx.vfs_file_id(), edit);
        },
    )
}

// Parses an if-else-if chain to get the conditions and the then branches until we encounter an else
// branch or the end.
fn parse_if_chain(if_expr: IfExpr) -> Option<(Vec<(Expr, BlockExpr)>, Option<BlockExpr>)> {
    let mut conds_blocks = Vec::new();
    let mut curr_if = if_expr;
    let tail = loop {
        let cond = curr_if.condition()?;
        conds_blocks.push((cond, curr_if.then_branch()?));
        match curr_if.else_branch() {
            Some(ElseBranch::IfExpr(e)) => {
                curr_if = e;
            }
            Some(ElseBranch::Block(b)) => {
                break Some(b);
            }
            None => break None,
        }
    };
    Some((conds_blocks, tail))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn move_guard_to_arm_body_range() {
        cov_mark::check!(move_guard_inapplicable_in_arm_body);
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
    fn move_non_naked_arm_cond_to_guard() {
        cov_mark::check!(move_guard_non_naked_if);
        check_assist_not_applicable(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        _ => {
            let cond = true;
            $0if cond {
                foo()
            }
        },
        _ => true
    }
}
"#,
        );
        check_assist_not_applicable(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        _ => {
            let cond = true;
            $0if cond {
                foo()
            } else {
                bar()
            }
        },
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
    fn move_guard_to_block_arm_body_works() {
        check_assist(
            move_guard_to_arm_body,
            r#"
fn main() {
    match 92 {
        x $0if x > 10 => {
            let _ = true;
            false
        },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x => if x > 10 {
            let _ = true;
            false
        },
        _ => true
    }
}
"#,
        );
    }

    #[test]
    fn move_let_guard_to_arm_body_works() {
        check_assist(
            move_guard_to_arm_body,
            r#"
fn main() {
    match 92 {
        x $0if (let 1 = x) => false,
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x => if (let 1 = x) {
            false
        },
        _ => true
    }
}
"#,
        );
    }

    #[test]
    fn move_multiline_guard_to_arm_body_works() {
        check_assist(
            move_guard_to_arm_body,
            r#"
fn main() {
    match 92 {
        x $0if true
            && true
            && true =>
        {
            {
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
        x => if true
            && true
            && true {
            {
                false
            }
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
        x => if x > 10$0 { false },
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
        cov_mark::check!(move_guard_ifelse_has_wildcard);
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
    fn move_arm_cond_in_block_to_match_guard_no_wildcard_works() {
        cov_mark::check_count!(move_guard_ifelse_has_wildcard, 0);
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
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
        x => {}
    }
}
"#,
        );
    }

    #[test]
    fn move_arm_cond_in_block_to_match_guard_wildcard_guard_works() {
        cov_mark::check_count!(move_guard_ifelse_has_wildcard, 0);
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
        _ if x > 10 => true,
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => false,
        x => {}
        _ if x > 10 => true,
    }
}
"#,
        );
    }

    #[test]
    fn move_arm_cond_in_block_to_match_guard_add_comma_works() {
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
    fn move_arm_cond_to_match_guard_if_let_works() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        x => if let 62 = x $0&& true { false },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if let 62 = x && true => false,
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
        x => if x $0> 10 {  },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => {  }
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
        x => if$0 x > 10 {
            92;
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
        }
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
            if x > $010 {
                92;
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
        x => if x > $010 {
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
            if x $0> 10 {
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
        x => if x > $010 {  } else { },
        _ => true
    }
}
"#,
            r#"
fn main() {
    match 92 {
        x if x > 10 => {  }
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
        x => if$0 x > 10 {
            92;
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
        x => if x $0> 10 {
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
            if x > $010 {
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
            if x > $010 {
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
        x => if x > $010 {
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

    #[test]
    fn move_arm_cond_to_match_guard_elseif() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        3 => true,
        x => if x $0> 10 {
            false
        } else if x > 5 {
            true
        } else if x > 4 {
            false
        } else {
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
        x if x > 5 => true,
        x if x > 4 => false,
        x => true,
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_elseif_in_block() {
        cov_mark::check!(move_guard_ifelse_in_block);
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        3 => true,
        x => {
            if x > $010 {
                false
            } else if x > 5 {
                true
            } else if x > 4 {
                false
            } else {
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
        x if x > 5 => true,
        x if x > 4 => false,
        x => true,
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_elseif_chain() {
        cov_mark::check!(move_guard_ifelse_else_tail);
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        3 => 0,
        x => if x $0> 10 {
            1
        } else if x > 5 {
            2
        } else if x > 3 {
            42;
            3
        } else {
            4
        },
    }
}
"#,
            r#"
fn main() {
    match 92 {
        3 => 0,
        x if x > 10 => 1,
        x if x > 5 => 2,
        x if x > 3 => {
            42;
            3
        }
        x => 4,
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_elseif_iflet() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        3 => 0,
        x => if x $0> 10 {
            1
        } else if x > 5 {
            2
        } else if let 4 = 4 {
            42;
            3
        } else {
            4
        },
    }
}"#,
            r#"
fn main() {
    match 92 {
        3 => 0,
        x if x > 10 => 1,
        x if x > 5 => 2,
        x if let 4 = 4 => {
            42;
            3
        }
        x => 4,
    }
}"#,
        );
    }

    #[test]
    fn move_arm_cond_to_match_guard_elseif_notail() {
        cov_mark::check!(move_guard_ifelse_notail);
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    match 92 {
        3 => 0,
        x => if x > $010 {
            1
        } else if x > 5 {
            2
        } else if x > 4 {
            42;
            3
        },
    }
}
"#,
            r#"
fn main() {
    match 92 {
        3 => 0,
        x if x > 10 => 1,
        x if x > 5 => 2,
        x if x > 4 => {
            42;
            3
        }
        x => {}
    }
}
"#,
        )
    }

    #[test]
    fn move_arm_cond_to_match_guard_elseif_exist_guard() {
        check_assist(
            move_arm_cond_to_match_guard,
            r#"
fn main() {
    let cond = true;
    match 92 {
        3 => true,
        x if cond => if x $0> 10 {
            false
        } else if x > 5 {
            true
        } else if x > 4 || x < -2 {
            false
        } else {
            true
        },
    }
}
"#,
            r#"
fn main() {
    let cond = true;
    match 92 {
        3 => true,
        x if cond && x > 10 => false,
        x if cond && x > 5 => true,
        x if cond && (x > 4 || x < -2) => false,
        x if cond => true,
    }
}
"#,
        )
    }
}
