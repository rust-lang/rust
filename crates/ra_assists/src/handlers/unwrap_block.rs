use ra_fmt::unwrap_trivial_block;
use ra_syntax::{
    ast::{self, ElseBranch, Expr, LoopBodyOwner},
    match_ast, AstNode, TextRange, T,
};

use crate::{AssistContext, AssistId, Assists};

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
    let assist_id = AssistId("unwrap_block");
    let assist_label = "Unwrap block";

    let (expr, expr_to_unwrap) = match_ast! {
        match parent {
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
            ast::IfExpr(if_expr) => {
                let mut resp = None;

                let then_branch = if_expr.then_branch()?;
                if then_branch.l_curly_token()?.text_range().contains_range(ctx.frange.range) {
                    if let Some(ancestor) = if_expr.syntax().parent().and_then(ast::IfExpr::cast) {
                        // For `else if` blocks
                        let ancestor_then_branch = ancestor.then_branch()?;
                        let l_curly_token = then_branch.l_curly_token()?;

                        let target = then_branch.syntax().text_range();
                        return acc.add(assist_id, assist_label, target, |edit| {
                            let range_to_del_else_if = TextRange::new(ancestor_then_branch.syntax().text_range().end(), l_curly_token.text_range().start());
                            let range_to_del_rest = TextRange::new(then_branch.syntax().text_range().end(), if_expr.syntax().text_range().end());

                            edit.set_cursor(ancestor_then_branch.syntax().text_range().end());
                            edit.delete(range_to_del_rest);
                            edit.delete(range_to_del_else_if);
                            edit.replace(target, update_expr_string(then_branch.to_string(), &[' ', '{']));
                        });
                    } else {
                        resp = Some((ast::Expr::IfExpr(if_expr.clone()), Expr::BlockExpr(then_branch)));
                    }
                } else if let Some(else_branch) = if_expr.else_branch() {
                    match else_branch {
                        ElseBranch::Block(else_block) => {
                            let l_curly_token = else_block.l_curly_token()?;
                            if l_curly_token.text_range().contains_range(ctx.frange.range) {
                                let target = else_block.syntax().text_range();
                                return acc.add(assist_id, assist_label, target, |edit| {
                                    let range_to_del = TextRange::new(then_branch.syntax().text_range().end(), l_curly_token.text_range().start());

                                    edit.set_cursor(then_branch.syntax().text_range().end());
                                    edit.delete(range_to_del);
                                    edit.replace(target, update_expr_string(else_block.to_string(), &[' ', '{']));
                                });
                            }
                        },
                        ElseBranch::IfExpr(_) => {},
                    }
                }

                resp?
            },
            _ => return None,
        }
    };

    let target = expr_to_unwrap.syntax().text_range();
    acc.add(assist_id, assist_label, target, |edit| {
        edit.set_cursor(expr.syntax().text_range().start());

        edit.replace(
            expr.syntax().text_range(),
            update_expr_string(expr_to_unwrap.to_string(), &[' ', '{', '\n']),
        );
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

fn update_expr_string(expr_str: String, trim_start_pat: &[char]) -> String {
    let expr_string = expr_str.trim_start_matches(trim_start_pat);
    let mut expr_string_lines: Vec<&str> = expr_string.lines().collect();
    expr_string_lines.pop(); // Delete last line

    expr_string_lines
        .into_iter()
        .map(|line| line.replacen("    ", "", 1)) // Delete indentation
        .collect::<Vec<String>>()
        .join("\n")
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
                if true {
                    foo();

                    //comment
                    bar();
                }<|>
                println!("bar");
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
                if true {
                    println!("true");

                    //comment
                    //bar();
                }<|>
                println!("bar");
            }
            "#,
        );
    }

    #[test]
    fn simple_if_else_if_nested() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                //bar();
                if true {
                    println!("true");

                    //comment
                    //bar();
                } else if false {
                    println!("bar");
                } else if true {<|>
                    println!("foo");
                }
            }
            "#,
            r#"
            fn main() {
                //bar();
                if true {
                    println!("true");

                    //comment
                    //bar();
                } else if false {
                    println!("bar");
                }<|>
                println!("foo");
            }
            "#,
        );
    }

    #[test]
    fn simple_if_else_if_nested_else() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                //bar();
                if true {
                    println!("true");

                    //comment
                    //bar();
                } else if false {
                    println!("bar");
                } else if true {
                    println!("foo");
                } else {<|>
                    println!("else");
                }
            }
            "#,
            r#"
            fn main() {
                //bar();
                if true {
                    println!("true");

                    //comment
                    //bar();
                } else if false {
                    println!("bar");
                } else if true {
                    println!("foo");
                }<|>
                println!("else");
            }
            "#,
        );
    }

    #[test]
    fn simple_if_else_if_nested_middle() {
        check_assist(
            unwrap_block,
            r#"
            fn main() {
                //bar();
                if true {
                    println!("true");

                    //comment
                    //bar();
                } else if false {
                    println!("bar");
                } else if true {<|>
                    println!("foo");
                } else {
                    println!("else");
                }
            }
            "#,
            r#"
            fn main() {
                //bar();
                if true {
                    println!("true");

                    //comment
                    //bar();
                } else if false {
                    println!("bar");
                }<|>
                println!("foo");
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
