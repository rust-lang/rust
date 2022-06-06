use ide_db::syntax_helpers::node_ext::for_each_break_and_continue_expr;
use syntax::ast::{self, AstNode, HasLoopBody};
use syntax::T;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: add_lifetime_to_type
//
// Adds a new lifetime to a struct, enum or union.
//
// ```
// struct Point {
//     x: &u32,
//     y: u32,
// }
// ```
// ->
// ```
// struct Point<'a> {
//     x: &'a u32,
//     y: u32,
// }
// ```
pub(crate) fn add_label_to_loop(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let loop_expr = ctx.find_node_at_offset::<ast::LoopExpr>()?;
    let loop_body = loop_expr.loop_body().and_then(|it| it.stmt_list());
    let mut related_exprs = vec![];
    related_exprs.push(ast::Expr::LoopExpr(loop_expr.clone()));
    for_each_break_and_continue_expr(loop_expr.label(), loop_body, &mut |expr| {
        if let ast::Expr::BreakExpr(_) | ast::Expr::ContinueExpr(_) = expr {
            related_exprs.push(expr)
        }
    });
    dbg!(loop_expr.syntax().text_range());

    acc.add(
        AssistId("add_label_to_loop", AssistKind::Generate),
        "Add Label",
        loop_expr.syntax().text_range(),
        |builder| {
            for expr in related_exprs {
                match expr {
                    ast::Expr::BreakExpr(break_expr) => {
                        if let Some(break_token) = break_expr.break_token() {
                            builder.insert(break_token.text_range().end(), " 'loop")
                        }
                    },
                    ast::Expr::ContinueExpr(continue_expr) => {
                        if let Some(continue_token) = continue_expr.continue_token() {
                            builder.insert(continue_token.text_range().end(), " 'loop")
                        }
                    },
                    ast::Expr::LoopExpr(loop_expr) => {
                        if let Some(loop_token) = loop_expr.loop_token() {
                            builder.insert(loop_token.text_range().start(), "'loop: ")
                        }
                    },
                    _ => todo!()
                }
            }
        },
    )
}


#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn add_label() {
        check_assist(
            add_label_to_loop,
            r#"
fn main() {
    loop$0 { 
        break; 
        continue; 
    }
}"#,
            r#"
fn main() {
    'loop: loop { 
        break 'loop; 
        continue 'loop; 
    }
}"#,
        );
    }

}
