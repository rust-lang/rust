use std::iter;

use ide_db::syntax_helpers::node_ext::is_pattern_cond;
use syntax::{
    AstNode, T,
    ast::{
        self, HasLoopBody,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    syntax_editor::{Element, Position},
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
    utils::invert_boolean_expression_legacy,
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
    let l_curly = while_body.stmt_list()?.l_curly_token()?;

    let target = while_expr.syntax().text_range();
    acc.add(
        AssistId::refactor_rewrite("convert_while_to_loop"),
        "Convert while to loop",
        target,
        |builder| {
            let mut edit = builder.make_editor(while_expr.syntax());
            let while_indent_level = IndentLevel::from_node(while_expr.syntax());

            let break_block = make::block_expr(
                iter::once(make::expr_stmt(make::expr_break(None, None)).into()),
                None,
            )
            .indent(IndentLevel(1));

            edit.replace_all(
                while_kw.syntax_element()..=while_cond.syntax().syntax_element(),
                vec![make::token(T![loop]).syntax_element()],
            );

            if is_pattern_cond(while_cond.clone()) {
                let then_branch = while_body.reset_indent().indent(IndentLevel(1));
                let if_expr = make::expr_if(while_cond, then_branch, Some(break_block.into()));
                let stmts = iter::once(make::expr_stmt(if_expr.into()).into());
                let block_expr = make::block_expr(stmts, None);
                edit.replace(while_body.syntax(), block_expr.indent(while_indent_level).syntax());
            } else {
                let if_cond = invert_boolean_expression_legacy(while_cond);
                let if_expr = make::expr_if(if_cond, break_block, None).indent(while_indent_level);
                if !while_body.syntax().text().contains_char('\n') {
                    edit.insert(
                        Position::after(&l_curly),
                        make::tokens::whitespace(&format!("\n{while_indent_level}")),
                    );
                }
                edit.insert_all(
                    Position::after(&l_curly),
                    vec![
                        make::tokens::whitespace(&format!("\n{}", while_indent_level + 1)).into(),
                        if_expr.syntax().syntax_element(),
                    ],
                );
            };

            builder.add_file_edits(ctx.vfs_file_id(), edit);
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
    fn convert_with_label() {
        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    'x: while$0 cond {
        foo();
        break 'x
    }
}
"#,
            r#"
fn main() {
    'x: loop {
        if !cond {
            break;
        }
        foo();
        break 'x
    }
}
"#,
        );

        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    'x: while$0 let Some(x) = cond {
        foo();
        break 'x
    }
}
"#,
            r#"
fn main() {
    'x: loop {
        if let Some(x) = cond {
            foo();
            break 'x
        } else {
            break;
        }
    }
}
"#,
        );
    }

    #[test]
    fn convert_with_attributes() {
        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    #[allow(unused)]
    while$0 cond {
        foo();
        break 'x
    }
}
"#,
            r#"
fn main() {
    #[allow(unused)]
    loop {
        if !cond {
            break;
        }
        foo();
        break 'x
    }
}
"#,
        );

        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    #[allow(unused)]
    #[deny(unsafe_code)]
    while$0 let Some(x) = cond {
        foo();
    }
}
"#,
            r#"
fn main() {
    #[allow(unused)]
    #[deny(unsafe_code)]
    loop {
        if let Some(x) = cond {
            foo();
        } else {
            break;
        }
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
    fn indentation() {
        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    {
        {
            while$0 cond {
                foo(
                    "xxx",
                );
            }
        }
    }
}
"#,
            r#"
fn main() {
    {
        {
            loop {
                if !cond {
                    break;
                }
                foo(
                    "xxx",
                );
            }
        }
    }
}
"#,
        );

        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    {
        {
            while$0 let Some(_) = foo() {
                bar(
                    "xxx",
                );
            }
        }
    }
}
"#,
            r#"
fn main() {
    {
        {
            loop {
                if let Some(_) = foo() {
                    bar(
                        "xxx",
                    );
                } else {
                    break;
                }
            }
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

    #[test]
    fn preserve_comments() {
        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    let mut i = 0;

    $0while i < 5 {
        // comment 1
        dbg!(i);
        // comment 2
        i += 1;
        // comment 3
    }
}
"#,
            r#"
fn main() {
    let mut i = 0;

    loop {
        if i >= 5 {
            break;
        }
        // comment 1
        dbg!(i);
        // comment 2
        i += 1;
        // comment 3
    }
}
"#,
        );

        check_assist(
            convert_while_to_loop,
            r#"
fn main() {
    let v = vec![1, 2, 3];
    let iter = v.iter();

    $0while let Some(i) = iter.next() {
        // comment 1
        dbg!(i);
        // comment 2
    }
}
"#,
            r#"
fn main() {
    let v = vec![1, 2, 3];
    let iter = v.iter();

    loop {
        if let Some(i) = iter.next() {
            // comment 1
            dbg!(i);
            // comment 2
        } else {
            break;
        }
    }
}
"#,
        );
    }
}
