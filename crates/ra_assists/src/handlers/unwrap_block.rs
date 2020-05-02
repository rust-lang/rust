use crate::{Assist, AssistCtx, AssistId};

use ast::{BlockExpr, Expr, ForExpr, IfExpr, LoopBodyOwner, LoopExpr, WhileExpr};
use ra_fmt::unwrap_trivial_block;
use ra_syntax::{ast, AstNode, TextRange, T};

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
pub(crate) fn unwrap_block(ctx: AssistCtx) -> Option<Assist> {
    let l_curly_token = ctx.find_token_at_offset(T!['{'])?;

    let res = if let Some(if_expr) = l_curly_token.ancestors().find_map(IfExpr::cast) {
        // if expression
        let expr_to_unwrap = if_expr.blocks().find_map(|expr| extract_expr(ctx.frange.range, expr));
        let expr_to_unwrap = expr_to_unwrap?;
        // Find if we are in a else if block
        let ancestor = if_expr.syntax().ancestors().skip(1).find_map(ast::IfExpr::cast);

        if let Some(ancestor) = ancestor {
            Some((ast::Expr::IfExpr(ancestor), expr_to_unwrap))
        } else {
            Some((ast::Expr::IfExpr(if_expr), expr_to_unwrap))
        }
    } else if let Some(for_expr) = l_curly_token.ancestors().find_map(ForExpr::cast) {
        // for expression
        let block_expr = for_expr.loop_body()?;
        extract_expr(ctx.frange.range, block_expr)
            .map(|expr_to_unwrap| (ast::Expr::ForExpr(for_expr), expr_to_unwrap))
    } else if let Some(while_expr) = l_curly_token.ancestors().find_map(WhileExpr::cast) {
        // while expression
        let block_expr = while_expr.loop_body()?;
        extract_expr(ctx.frange.range, block_expr)
            .map(|expr_to_unwrap| (ast::Expr::WhileExpr(while_expr), expr_to_unwrap))
    } else if let Some(loop_expr) = l_curly_token.ancestors().find_map(LoopExpr::cast) {
        // loop expression
        let block_expr = loop_expr.loop_body()?;
        extract_expr(ctx.frange.range, block_expr)
            .map(|expr_to_unwrap| (ast::Expr::LoopExpr(loop_expr), expr_to_unwrap))
    } else {
        None
    };

    let (expr, expr_to_unwrap) = res?;
    ctx.add_assist(AssistId("unwrap_block"), "Unwrap block", |edit| {
        edit.set_cursor(expr.syntax().text_range().start());
        edit.target(expr_to_unwrap.syntax().text_range());

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

fn extract_expr(cursor_range: TextRange, block_expr: BlockExpr) -> Option<Expr> {
    let block = block_expr.block()?;
    let cursor_in_range = block.l_curly_token()?.text_range().contains_range(cursor_range);

    if cursor_in_range {
        Some(unwrap_trivial_block(block_expr))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable};

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
