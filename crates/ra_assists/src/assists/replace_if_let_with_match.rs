use format_buf::format;
use hir::db::HirDatabase;
use ra_fmt::extract_trivial_expression;
use ra_syntax::{ast, AstNode};

use crate::{Assist, AssistCtx, AssistId};

// Assist: replace_if_let_with_match
//
// Replaces `if let` with an else branch with a `match` expression.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     <|>if let Action::Move { distance } = action {
//         foo(distance)
//     } else {
//         bar()
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } => foo(distance),
//         _ => bar(),
//     }
// }
// ```
pub(crate) fn replace_if_let_with_match(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let if_expr: ast::IfExpr = ctx.find_node_at_offset()?;
    let cond = if_expr.condition()?;
    let pat = cond.pat()?;
    let expr = cond.expr()?;
    let then_block = if_expr.then_branch()?;
    let else_block = match if_expr.else_branch()? {
        ast::ElseBranch::Block(it) => it,
        ast::ElseBranch::IfExpr(_) => return None,
    };

    ctx.add_assist(AssistId("replace_if_let_with_match"), "replace with match", |edit| {
        let match_expr = build_match_expr(expr, pat, then_block, else_block);
        edit.target(if_expr.syntax().text_range());
        edit.replace_node_and_indent(if_expr.syntax(), match_expr);
        edit.set_cursor(if_expr.syntax().text_range().start())
    })
}

fn build_match_expr(
    expr: ast::Expr,
    pat1: ast::Pat,
    arm1: ast::BlockExpr,
    arm2: ast::BlockExpr,
) -> String {
    let mut buf = String::new();
    format!(buf, "match {} {{\n", expr.syntax().text());
    format!(buf, "    {} => {}\n", pat1.syntax().text(), format_arm(&arm1));
    format!(buf, "    _ => {}\n", format_arm(&arm2));
    buf.push_str("}");
    buf
}

fn format_arm(block: &ast::BlockExpr) -> String {
    match extract_trivial_expression(block) {
        Some(e) if !e.syntax().text().contains_char('\n') => format!("{},", e.syntax().text()),
        _ => block.syntax().text().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_target};

    #[test]
    fn test_replace_if_let_with_match_unwraps_simple_expressions() {
        check_assist(
            replace_if_let_with_match,
            "
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if <|>let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
}           ",
            "
impl VariantData {
    pub fn is_struct(&self) -> bool {
        <|>match *self {
            VariantData::Struct(..) => true,
            _ => false,
        }
    }
}           ",
        )
    }

    #[test]
    fn test_replace_if_let_with_match_doesnt_unwrap_multiline_expressions() {
        check_assist(
            replace_if_let_with_match,
            "
fn foo() {
    if <|>let VariantData::Struct(..) = a {
        bar(
            123
        )
    } else {
        false
    }
}           ",
            "
fn foo() {
    <|>match a {
        VariantData::Struct(..) => {
            bar(
                123
            )
        }
        _ => false,
    }
}           ",
        )
    }

    #[test]
    fn replace_if_let_with_match_target() {
        check_assist_target(
            replace_if_let_with_match,
            "
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if <|>let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
}           ",
            "if let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }",
        );
    }
}
