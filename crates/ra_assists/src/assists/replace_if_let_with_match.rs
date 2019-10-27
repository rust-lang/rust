//! FIXME: write short doc here

use format_buf::format;
use hir::db::HirDatabase;
use ra_fmt::extract_trivial_expression;
use ra_syntax::{ast, AstNode};

use crate::{Assist, AssistCtx, AssistId};

pub(crate) fn replace_if_let_with_match(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let if_expr: ast::IfExpr = ctx.find_node_at_offset()?;
    let cond = if_expr.condition()?;
    let pat = cond.pat()?;
    let expr = cond.expr()?;
    let then_block = if_expr.then_branch()?;
    let else_block = match if_expr.else_branch()? {
        ast::ElseBranch::Block(it) => it,
        ast::ElseBranch::IfExpr(_) => return None,
    };

    ctx.add_action(AssistId("replace_if_let_with_match"), "replace with match", |edit| {
        let match_expr = build_match_expr(expr, pat, then_block, else_block);
        edit.target(if_expr.syntax().text_range());
        edit.replace_node_and_indent(if_expr.syntax(), match_expr);
        edit.set_cursor(if_expr.syntax().text_range().start())
    });

    ctx.build()
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
        None => block.syntax().text().to_string(),
        Some(e) => format!("{},", e.syntax().text()),
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
