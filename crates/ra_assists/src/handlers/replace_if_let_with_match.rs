use ra_fmt::unwrap_trivial_block;
use ra_syntax::{
    ast::{self, make},
    AstNode,
};

use crate::{Assist, AssistCtx, AssistId};
use ast::edit::IndentLevel;

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
pub(crate) fn replace_if_let_with_match(ctx: AssistCtx) -> Option<Assist> {
    let if_expr: ast::IfExpr = ctx.find_node_at_offset()?;
    let cond = if_expr.condition()?;
    let pat = cond.pat()?;
    let expr = cond.expr()?;
    let then_block = if_expr.then_branch()?;
    let else_block = match if_expr.else_branch()? {
        ast::ElseBranch::Block(it) => it,
        ast::ElseBranch::IfExpr(_) => return None,
    };

    ctx.add_assist(AssistId("replace_if_let_with_match"), "Replace with match", |edit| {
        let match_expr = {
            let then_arm = {
                let then_expr = unwrap_trivial_block(then_block);
                make::match_arm(vec![pat], then_expr)
            };
            let else_arm = {
                let else_expr = unwrap_trivial_block(else_block);
                make::match_arm(vec![make::placeholder_pat().into()], else_expr)
            };
            make::expr_match(expr, make::match_arm_list(vec![then_arm, else_arm]))
        };

        let match_expr = IndentLevel::from_node(if_expr.syntax()).increase_indent(match_expr);

        edit.target(if_expr.syntax().text_range());
        edit.set_cursor(if_expr.syntax().text_range().start());
        edit.replace_ast::<ast::Expr>(if_expr.into(), match_expr.into());
    })
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
