use crate::{Assist, AssistCtx, AssistId};

use ast::LoopBodyOwner;
use ra_fmt::unwrap_trivial_block;
use ra_syntax::{ast, AstNode};

// Assist: unwrap_block
//
// Removes the `mut` keyword.
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
//     <|>println!("foo");
// }
// ```
pub(crate) fn unwrap_block(ctx: AssistCtx) -> Option<Assist> {
    let res = if let Some(if_expr) = ctx.find_node_at_offset::<ast::IfExpr>() {
        // if expression
        let mut expr_to_unwrap: Option<ast::Expr> = None;
        for block_expr in if_expr.blocks() {
            if let Some(block) = block_expr.block() {
                let cursor_in_range =
                    block.l_curly_token()?.text_range().contains_range(ctx.frange.range);

                if cursor_in_range {
                    let exprto = unwrap_trivial_block(block_expr);
                    expr_to_unwrap = Some(exprto);
                    break;
                }
            }
        }
        let expr_to_unwrap = expr_to_unwrap?;
        // Find if we are in a else if block
        let ancestor = ctx
            .sema
            .ancestors_with_macros(if_expr.syntax().clone())
            .skip(1)
            .find_map(ast::IfExpr::cast);

        if let Some(ancestor) = ancestor {
            Some((ast::Expr::IfExpr(ancestor), expr_to_unwrap))
        } else {
            Some((ast::Expr::IfExpr(if_expr), expr_to_unwrap))
        }
    } else if let Some(for_expr) = ctx.find_node_at_offset::<ast::ForExpr>() {
        // for expression
        let block_expr = for_expr.loop_body()?;
        let block = block_expr.block()?;
        let cursor_in_range = block.l_curly_token()?.text_range().contains_range(ctx.frange.range);

        if cursor_in_range {
            let expr_to_unwrap = unwrap_trivial_block(block_expr);

            Some((ast::Expr::ForExpr(for_expr), expr_to_unwrap))
        } else {
            None
        }
    } else if let Some(while_expr) = ctx.find_node_at_offset::<ast::WhileExpr>() {
        // while expression
        let block_expr = while_expr.loop_body()?;
        let block = block_expr.block()?;
        let cursor_in_range = block.l_curly_token()?.text_range().contains_range(ctx.frange.range);

        if cursor_in_range {
            let expr_to_unwrap = unwrap_trivial_block(block_expr);

            Some((ast::Expr::WhileExpr(while_expr), expr_to_unwrap))
        } else {
            None
        }
    } else if let Some(loop_expr) = ctx.find_node_at_offset::<ast::LoopExpr>() {
        // loop expression
        let block_expr = loop_expr.loop_body()?;
        let block = block_expr.block()?;
        let cursor_in_range = block.l_curly_token()?.text_range().contains_range(ctx.frange.range);

        if cursor_in_range {
            let expr_to_unwrap = unwrap_trivial_block(block_expr);

            Some((ast::Expr::LoopExpr(loop_expr), expr_to_unwrap))
        } else {
            None
        }
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
    fn issue_example_with_if() {
        check_assist(
            unwrap_block,
            r#"
            fn complete_enum_variants(acc: &mut Completions, ctx: &CompletionContext, ty: &Type) {
                if let Some(ty) = &ctx.expected_type {<|>
                    if let Some(Adt::Enum(enum_data)) = ty.as_adt() {
                        let variants = enum_data.variants(ctx.db);

                        let module = if let Some(module) = ctx.scope().module() {
                            // Compute path from the completion site if available.
                            module
                        } else {
                            // Otherwise fall back to the enum's definition site.
                            enum_data.module(ctx.db)
                        };

                        for variant in variants {
                            if let Some(path) = module.find_use_path(ctx.db, ModuleDef::from(variant)) {
                                // Variants with trivial paths are already added by the existing completion logic,
                                // so we should avoid adding these twice
                                if path.segments.len() > 1 {
                                    acc.add_enum_variant(ctx, variant, Some(path.to_string()));
                                }
                            }
                        }
                    }
                }
            }
            "#,
            r#"
            fn complete_enum_variants(acc: &mut Completions, ctx: &CompletionContext, ty: &Type) {
                <|>if let Some(Adt::Enum(enum_data)) = ty.as_adt() {
                    let variants = enum_data.variants(ctx.db);

                    let module = if let Some(module) = ctx.scope().module() {
                        // Compute path from the completion site if available.
                        module
                    } else {
                        // Otherwise fall back to the enum's definition site.
                        enum_data.module(ctx.db)
                    };

                    for variant in variants {
                        if let Some(path) = module.find_use_path(ctx.db, ModuleDef::from(variant)) {
                            // Variants with trivial paths are already added by the existing completion logic,
                            // so we should avoid adding these twice
                            if path.segments.len() > 1 {
                                acc.add_enum_variant(ctx, variant, Some(path.to_string()));
                            }
                        }
                    }
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
