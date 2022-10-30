use std::iter::once;

use ide_db::syntax_helpers::node_ext::is_pattern_cond;
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make, HasLoopBody,
    },
    AstNode, T,
};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::invert_boolean_expression,
    AssistId, AssistKind,
};

// Assist: convert_while_to_loop
//
// Replace a while with a loop.
//
// ```
// fn main() {
//     $0while cond {
//         foo();
//     }
// }
// ```
// ->
// ```
// fn main() {
//     loop {
//         if !cond {
//             break;
//         }
//         foo();
//     }
// }
// ```
pub(crate) fn convert_while_to_loop(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let while_kw = ctx.find_token_syntax_at_offset(T![while])?;
    let while_expr = while_kw.parent().and_then(ast::WhileExpr::cast)?;
    let while_body = while_expr.loop_body()?;
    let while_cond = while_expr.condition()?;

    let target = while_expr.syntax().text_range();
    acc.add(
        AssistId("convert_while_to_loop", AssistKind::RefactorRewrite),
        "Convert while to loop",
        target,
        |edit| {
            let while_indent_level = IndentLevel::from_node(while_expr.syntax());

            let break_block =
                make::block_expr(once(make::expr_stmt(make::expr_break(None, None)).into()), None)
                    .indent(while_indent_level);
            let block_expr = if is_pattern_cond(while_cond.clone()) {
                let if_expr = make::expr_if(while_cond, while_body, Some(break_block.into()));
                let stmts = once(make::expr_stmt(if_expr).into());
                make::block_expr(stmts, None)
            } else {
                let if_cond = invert_boolean_expression(while_cond);
                let if_expr = make::expr_if(if_cond, break_block, None);
                let stmts = once(make::expr_stmt(if_expr).into()).chain(while_body.statements());
                make::block_expr(stmts, while_body.tail_expr())
            };

            let replacement = make::expr_loop(block_expr.indent(while_indent_level));
            edit.replace(target, replacement.syntax().text())
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn convert_inside_fn() {
        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    while$0 cond {
        foo();
    }
}
"#,
            r#"
fn main() {
    loop {
        if !cond {
            break;
        }
        foo();
    }
}
"#,
        );
    }

    #[test]
    fn convert_busy_wait() {
        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    while$0 cond() {}
}
"#,
            r#"
fn main() {
    loop {
        if !cond() {
            break;
        }
    }
}
"#,
        );
    }

    #[test]
    fn convert_trailing_expr() {
        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    while$0 cond() {
        bar()
    }
}
"#,
            r#"
fn main() {
    loop {
        if !cond() {
            break;
        }
        bar()
    }
}
"#,
        );
    }

    #[test]
    fn convert_while_let() {
        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    while$0 let Some(_) = foo() {
        bar();
    }
}
"#,
            r#"
fn main() {
    loop {
        if let Some(_) = foo() {
            bar();
        } else {
            break;
        }
    }
}
"#,
        );
    }

    #[test]
    fn ignore_cursor_in_body() {
        check_assist_not_applicable(
            convert_while_to_loop,
            r#"
fn main() {
    while cond {$0
        bar();
    }
}
"#,
        );
    }
}
