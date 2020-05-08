use crate::{AssistContext, AssistId, Assists};

use ast::LoopBodyOwner;
use ra_fmt::unwrap_trivial_block;
use ra_syntax::{ast, match_ast, AstNode, TextRange, T};

// Assist: unwrap_block
//
// This assist removes if...else, for, while and loop control statements to just keep the body.
//
// ```
// fn foo() {
//     if true {<|>
//         println!("foo");
//     }
// }
// ```
// ->
// ```
// fn foo() {
//     println!("foo");
// }
// ```
pub(crate) fn unwrap_block(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let l_curly_token = ctx.find_token_at_offset(T!['{'])?;
    let block = ast::BlockExpr::cast(l_curly_token.parent())?;
    let parent = block.syntax().parent()?;
    let (expr, expr_to_unwrap) = match_ast! {
        match parent {
            ast::IfExpr(if_expr) => {
                let expr_to_unwrap = if_expr.blocks().find_map(|expr| extract_expr(ctx.frange.range, expr));
                let expr_to_unwrap = expr_to_unwrap?;
                // Find if we are in a else if block
                let ancestor = if_expr.syntax().parent().and_then(ast::IfExpr::cast);

                match ancestor {
                    None => (ast::Expr::IfExpr(if_expr), expr_to_unwrap),
                    Some(ancestor) => (ast::Expr::IfExpr(ancestor), expr_to_unwrap),
                }
            },
            ast::ForExpr(for_expr) => {
                let block_expr = for_expr.loop_body()?;
                let expr_to_unwrap = extract_expr(ctx.frange.range, block_expr)?;
                (ast::Expr::ForExpr(for_expr), expr_to_unwrap)
            },
            ast::WhileExpr(while_expr) => {
                let block_expr = while_expr.loop_body()?;
                let expr_to_unwrap = extract_expr(ctx.frange.range, block_expr)?;
                (ast::Expr::WhileExpr(while_expr), expr_to_unwrap)
            },
            ast::LoopExpr(loop_expr) => {
                let block_expr = loop_expr.loop_body()?;
                let expr_to_unwrap = extract_expr(ctx.frange.range, block_expr)?;
                (ast::Expr::LoopExpr(loop_expr), expr_to_unwrap)
            },
            _ => return None,
        }
    };

    let target = expr_to_unwrap.syntax().text_range();
    acc.add(AssistId("unwrap_block"), "Unwrap block", target, |edit| {
        edit.set_cursor(expr.syntax().text_range().start());

        let pat_start: &[_] = &[' ', '{', '\n'];
        let expr_to_unwrap = expr_to_unwrap.to_string();
        let expr_string = expr_to_unwrap.trim_start_matches(pat_start);
        let mut expr_string_lines: Vec<&str> = expr_string.lines().collect();
        expr_string_lines.pop(); // Delete last line

        let expr_string = expr_string_lines
            .into_iter()
            .map(|line| line.replacen("    ", "", 1)) // Delete indentation
            .collect::<Vec<String>>()
            .join("\n");

        edit.replace(expr.syntax().text_range(), expr_string);
    })
}

fn extract_expr(cursor_range: TextRange, block: ast::BlockExpr) -> Option<ast::Expr> {
    let cursor_in_range = block.l_curly_token()?.text_range().contains_range(cursor_range);

    if cursor_in_range {
        Some(unwrap_trivial_block(block))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn simple_if() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                bar();
                if true {<|>
                    foo();

                    //comment
                    bar();
                } else {
                    println!("bar");
                }
            }
            "#,
            r#"
            fn main() {
                bar();
                <|>foo();

                //comment
                bar();
            }
            "#,
        );
    }

    #[test]
    fn simple_if_else() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                bar();
                if true {
                    foo();

                    //comment
                    bar();
                } else {<|>
                    println!("bar");
                }
            }
            "#,
            r#"
            fn main() {
                bar();
                <|>println!("bar");
            }
            "#,
        );
    }

    #[test]
    fn simple_if_else_if() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                //bar();
                if true {
                    println!("true");

                    //comment
                    //bar();
                } else if false {<|>
                    println!("bar");
                } else {
                    println!("foo");
                }
            }
            "#,
            r#"
            fn main() {
                //bar();
                <|>println!("bar");
            }
            "#,
        );
    }

    #[test]
    fn simple_if_bad_cursor_position() {
        check_assist_not_applicable(
            unwrap_block,
            r#"
            fn main() {
                bar();<|>
                if true {
                    foo();

                    //comment
                    bar();
                } else {
                    println!("bar");
                }
            }
            "#,
        );
    }

    #[test]
    fn simple_for() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                for i in 0..5 {<|>
                    if true {
                        foo();

                        //comment
                        bar();
                    } else {
                        println!("bar");
                    }
                }
            }
            "#,
            r#"
            fn main() {
                <|>if true {
                    foo();

                    //comment
                    bar();
                } else {
                    println!("bar");
                }
            }
            "#,
        );
    }

    #[test]
    fn simple_if_in_for() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                for i in 0..5 {
                    if true {<|>
                        foo();

                        //comment
                        bar();
                    } else {
                        println!("bar");
                    }
                }
            }
            "#,
            r#"
            fn main() {
                for i in 0..5 {
                    <|>foo();

                    //comment
                    bar();
                }
            }
            "#,
        );
    }

    #[test]
    fn simple_loop() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                loop {<|>
                    if true {
                        foo();

                        //comment
                        bar();
                    } else {
                        println!("bar");
                    }
                }
            }
            "#,
            r#"
            fn main() {
                <|>if true {
                    foo();

                    //comment
                    bar();
                } else {
                    println!("bar");
                }
            }
            "#,
        );
    }

    #[test]
    fn simple_while() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                while true {<|>
                    if true {
                        foo();

                        //comment
                        bar();
                    } else {
                        println!("bar");
                    }
                }
            }
            "#,
            r#"
            fn main() {
                <|>if true {
                    foo();

                    //comment
                    bar();
                } else {
                    println!("bar");
                }
            }
            "#,
        );
    }

    #[test]
    fn simple_if_in_while_bad_cursor_position() {
        check_assist_not_applicable(
            unwrap_block,
            r#"
            fn main() {
                while true {
                    if true {
                        foo();<|>

                        //comment
                        bar();
                    } else {
                        println!("bar");
                    }
                }
            }
            "#,
        );
    }
}
