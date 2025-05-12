use ide_db::syntax_helpers::node_ext::for_each_break_and_continue_expr;
use syntax::{
    T,
    ast::{self, AstNode, HasLoopBody},
};

use crate::{AssistContext, AssistId, Assists};

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
//     'l: loop {
//         break 'l;
//         continue 'l;
//     }
// }
// ```
pub(crate) fn add_label_to_loop(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let loop_kw = ctx.find_token_syntax_at_offset(T![loop])?;
    let loop_expr = loop_kw.parent().and_then(ast::LoopExpr::cast)?;
    if loop_expr.label().is_some() {
        return None;
    }

    acc.add(
        AssistId::generate("add_label_to_loop"),
        "Add Label",
        loop_expr.syntax().text_range(),
        |builder| {
            builder.insert(loop_kw.text_range().start(), "'l: ");

            let loop_body = loop_expr.loop_body().and_then(|it| it.stmt_list());
            for_each_break_and_continue_expr(
                loop_expr.label(),
                loop_body,
                &mut |expr| match expr {
                    ast::Expr::BreakExpr(break_expr) => {
                        if let Some(break_token) = break_expr.break_token() {
                            builder.insert(break_token.text_range().end(), " 'l")
                        }
                    }
                    ast::Expr::ContinueExpr(continue_expr) => {
                        if let Some(continue_token) = continue_expr.continue_token() {
                            builder.insert(continue_token.text_range().end(), " 'l")
                        }
                    }
                    _ => {}
                },
            );
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
    'l: loop {
        break 'l;
        continue 'l;
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
    'l: loop {
        break 'l;
        continue 'l;
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
        'l: loop {
            break 'l;
            continue 'l;
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
    'l: loop$0 {
        break 'l;
        continue 'l;
    }
}"#,
        );
    }
}
