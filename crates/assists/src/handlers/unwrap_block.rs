use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
    },
    AstNode, TextRange, T,
};

use crate::{utils::unwrap_trivial_block, AssistContext, AssistId, AssistKind, Assists};

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
    let assist_id = AssistId("unwrap_block", AssistKind::RefactorRewrite);
    let assist_label = "Unwrap block";

    let l_curly_token = ctx.find_token_at_offset(T!['{'])?;
    let mut block = ast::BlockExpr::cast(l_curly_token.parent())?;
    let mut parent = block.syntax().parent()?;
    if ast::MatchArm::can_cast(parent.kind()) {
        parent = parent.ancestors().find(|it| ast::MatchExpr::can_cast(it.kind()))?
    }

    let parent = ast::Expr::cast(parent)?;

    match parent.clone() {
        ast::Expr::ForExpr(_) | ast::Expr::WhileExpr(_) | ast::Expr::LoopExpr(_) => (),
        ast::Expr::MatchExpr(_) => block = block.dedent(IndentLevel(1)),
        ast::Expr::IfExpr(if_expr) => {
            let then_branch = if_expr.then_branch()?;
            if then_branch == block {
                if let Some(ancestor) = if_expr.syntax().parent().and_then(ast::IfExpr::cast) {
                    // For `else if` blocks
                    let ancestor_then_branch = ancestor.then_branch()?;

                    let target = then_branch.syntax().text_range();
                    return acc.add(assist_id, assist_label, target, |edit| {
                        let range_to_del_else_if = TextRange::new(
                            ancestor_then_branch.syntax().text_range().end(),
                            l_curly_token.text_range().start(),
                        );
                        let range_to_del_rest = TextRange::new(
                            then_branch.syntax().text_range().end(),
                            if_expr.syntax().text_range().end(),
                        );

                        edit.delete(range_to_del_rest);
                        edit.delete(range_to_del_else_if);
                        edit.replace(
                            target,
                            update_expr_string(then_branch.to_string(), &[' ', '{']),
                        );
                    });
                }
            } else {
                let target = block.syntax().text_range();
                return acc.add(assist_id, assist_label, target, |edit| {
                    let range_to_del = TextRange::new(
                        then_branch.syntax().text_range().end(),
                        l_curly_token.text_range().start(),
                    );

                    edit.delete(range_to_del);
                    edit.replace(target, update_expr_string(block.to_string(), &[' ', '{']));
                });
            }
        }
        _ => return None,
    };

    let unwrapped = unwrap_trivial_block(block);
    let target = unwrapped.syntax().text_range();
    acc.add(assist_id, assist_label, target, |builder| {
        builder.replace(
            parent.syntax().text_range(),
            update_expr_string(unwrapped.to_string(), &[' ', '{', '\n']),
        );
    })
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
                foo();

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
                }
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
                }
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
                }
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
                }
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
                }
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
                    foo();

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
    fn unwrap_match_arm() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) => {<|>
            let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
            Some((*id, rel_path))
        }
        Err(_) => None,
    }
}
"#,
            r#"
fn main() {
    let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
    Some((*id, rel_path))
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
