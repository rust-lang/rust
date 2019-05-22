use hir::db::HirDatabase;
use ra_syntax::{
    TextUnit,
    SyntaxElement,
    ast::{MatchArm, AstNode, AstToken, IfExpr},
    ast,
};

use crate::{AssistCtx, Assist, AssistId};

pub(crate) fn move_guard_to_arm_body(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let match_arm = ctx.node_at_offset::<MatchArm>()?;
    let guard = match_arm.guard()?;
    let space_before_guard = guard.syntax().prev_sibling_or_token();

    let guard_conditions = guard.expr()?;
    let arm_expr = match_arm.expr()?;
    let buf = format!("if {} {{ {} }}", guard_conditions.syntax().text(), arm_expr.syntax().text());

    ctx.add_action(AssistId("move_guard_to_arm_body"), "move guard to arm body", |edit| {
        edit.target(guard.syntax().range());
        let offseting_amount = match space_before_guard {
            Some(SyntaxElement::Token(tok)) => {
                if let Some(_) = ast::Whitespace::cast(tok) {
                    let ele = space_before_guard.unwrap().range();
                    edit.delete(ele);
                    ele.len()
                } else {
                    TextUnit::from(0)
                }
            }
            _ => TextUnit::from(0),
        };

        edit.delete(guard.syntax().range());
        edit.replace_node_and_indent(arm_expr.syntax(), buf);
        edit.set_cursor(arm_expr.syntax().range().start() + TextUnit::from(3) - offseting_amount);
    });
    ctx.build()
}

pub(crate) fn move_arm_cond_to_match_guard(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let match_arm: &MatchArm = ctx.node_at_offset::<MatchArm>()?;
    let last_match_pat = match_arm.pats().last()?;

    let arm_body = match_arm.expr()?;
    let if_expr: &IfExpr = IfExpr::cast(arm_body.syntax())?;
    let cond = if_expr.condition()?;
    let then_block = if_expr.then_branch()?;

    // Not support if with else branch
    if let Some(_) = if_expr.else_branch() {
        return None;
    }
    // Not support moving if let to arm guard
    if let Some(_) = cond.pat() {
        return None;
    }

    let buf = format!(" if {}", cond.syntax().text());

    ctx.add_action(
        AssistId("move_arm_cond_to_match_guard"),
        "move condition to match guard",
        |edit| {
            edit.target(if_expr.syntax().range());
            let then_only_expr = then_block.statements().next().is_none();

            match then_block.expr() {
                Some(then_expr) if then_only_expr => {
                    edit.replace(if_expr.syntax().range(), then_expr.syntax().text())
                }
                _ => edit.replace(if_expr.syntax().range(), then_block.syntax().text()),
            }

            edit.insert(last_match_pat.syntax().range().end(), buf);
            edit.set_cursor(last_match_pat.syntax().range().end() + TextUnit::from(1));
        },
    );
    ctx.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::helpers::{ check_assist, check_assist_target, check_assist_not_applicable };

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
                    '\r' => if chars.clone().next() == Some('\n') { <|>false },
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
                    y @ 4 | y @ 5 => if y > 5 { <|>true },
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
                    '\r' <|>if chars.clone().next() == Some('\n') => false,
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
                    '\r' <|>if chars.clone().next().is_some() => {  },
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
                    '\r' <|>if chars.clone().next().is_some() => {
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
