use ide_db::syntax_helpers::node_ext::for_each_break_and_continue_expr;
use syntax::ast::{self, AstNode, HasLoopBody};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: add_label_to_loop
//
// Adds a label to a loop.
//
// ```
// fn main() {
//     loop$0 {
//         break;
//         continue;
//     }
// }
// ```
// ->
// ```
// fn main() {
//     'loop: loop {
//         break 'loop;
//         continue 'loop;
//     }
// }
// ```
pub(crate) fn add_label_to_loop(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let loop_expr = ctx.find_node_at_offset::<ast::LoopExpr>()?;
    if loop_expr.label().is_some() {
        return None;
    }
    let loop_body = loop_expr.loop_body().and_then(|it| it.stmt_list());
    let mut related_exprs = vec![];
    related_exprs.push(ast::Expr::LoopExpr(loop_expr.clone()));
    for_each_break_and_continue_expr(loop_expr.label(), loop_body, &mut |expr| {
        if let ast::Expr::BreakExpr(_) | ast::Expr::ContinueExpr(_) = expr {
            related_exprs.push(expr)
        }
    });

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
                    }
                    ast::Expr::ContinueExpr(continue_expr) => {
                        if let Some(continue_token) = continue_expr.continue_token() {
                            builder.insert(continue_token.text_range().end(), " 'loop")
                        }
                    }
                    ast::Expr::LoopExpr(loop_expr) => {
                        if let Some(loop_token) = loop_expr.loop_token() {
                            builder.insert(loop_token.text_range().start(), "'loop: ")
                        }
                    }
                    _ => {}
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

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

    #[test]
    fn add_label_to_outer_loop() {
        check_assist(
            add_label_to_loop,
            r#"
fn main() {
    loop$0 {
        break;
        continue;
        loop {
            break;
            continue;
        }
    }
}"#,
            r#"
fn main() {
    'loop: loop {
        break 'loop;
        continue 'loop;
        loop {
            break;
            continue;
        }
    }
}"#,
        );
    }

    #[test]
    fn add_label_to_inner_loop() {
        check_assist(
            add_label_to_loop,
            r#"
fn main() {
    loop {
        break;
        continue;
        loop$0 {
            break;
            continue;
        }
    }
}"#,
            r#"
fn main() {
    loop {
        break;
        continue;
        'loop: loop {
            break 'loop;
            continue 'loop;
        }
    }
}"#,
        );
    }

    #[test]
    fn do_not_add_label_if_exists() {
        check_assist_not_applicable(
            add_label_to_loop,
            r#"
fn main() {
    'loop: loop$0 {
        break 'loop;
        continue 'loop;
    }
}"#,
        );
    }
}
