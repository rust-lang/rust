use std::{iter::once, ops::RangeInclusive};

use hir::db::HirDatabase;
use ra_syntax::{
    algo::replace_children,
    ast::{self, edit::IndentLevel, make, Block, Pat::TupleStructPat},
    AstNode,
    SyntaxKind::{FN_DEF, LOOP_EXPR, L_CURLY, R_CURLY, WHILE_EXPR, WHITESPACE},
    SyntaxNode,
};

use crate::{
    assist_ctx::{Assist, AssistCtx},
    AssistId,
};

// Assist: convert_to_guarded_return
//
// Replace a large conditional with a guarded return.
//
// ```
// fn main() {
//     <|>if cond {
//         foo();
//         bar();
//     }
// }
// ```
// ->
// ```
// fn main() {
//     if !cond {
//         return;
//     }
//     foo();
//     bar();
// }
// ```
pub(crate) fn convert_to_guarded_return(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let if_expr: ast::IfExpr = ctx.find_node_at_offset()?;
    if if_expr.else_branch().is_some() {
        return None;
    }

    let cond = if_expr.condition()?;

    // Check if there is an IfLet that we can handle.
    let if_let_pat = match cond.pat() {
        None => None, // No IfLet, supported.
        Some(TupleStructPat(pat)) if pat.args().count() == 1 => {
            let path = pat.path()?;
            match path.qualifier() {
                None => {
                    let bound_ident = pat.args().next().unwrap();
                    Some((path, bound_ident))
                }
                Some(_) => return None,
            }
        }
        Some(_) => return None, // Unsupported IfLet.
    };

    let cond_expr = cond.expr()?;
    let then_block = if_expr.then_branch()?.block()?;

    let parent_block = if_expr.syntax().parent()?.ancestors().find_map(ast::Block::cast)?;

    if parent_block.expr()? != if_expr.clone().into() {
        return None;
    }

    // check for early return and continue
    let first_in_then_block = then_block.syntax().first_child()?;
    if ast::ReturnExpr::can_cast(first_in_then_block.kind())
        || ast::ContinueExpr::can_cast(first_in_then_block.kind())
        || first_in_then_block
            .children()
            .any(|x| ast::ReturnExpr::can_cast(x.kind()) || ast::ContinueExpr::can_cast(x.kind()))
    {
        return None;
    }

    let parent_container = parent_block.syntax().parent()?.parent()?;

    let early_expression: ast::Expr = match parent_container.kind() {
        WHILE_EXPR | LOOP_EXPR => make::expr_continue(),
        FN_DEF => make::expr_return(),
        _ => return None,
    };

    if then_block.syntax().first_child_or_token().map(|t| t.kind() == L_CURLY).is_none() {
        return None;
    }

    then_block.syntax().last_child_or_token().filter(|t| t.kind() == R_CURLY)?;
    let cursor_position = ctx.frange.range.start();

    ctx.add_assist(AssistId("convert_to_guarded_return"), "convert to guarded return", |edit| {
        let if_indent_level = IndentLevel::from_node(&if_expr.syntax());
        let new_block = match if_let_pat {
            None => {
                // If.
                let early_expression = &(early_expression.syntax().to_string() + ";");
                let new_expr = if_indent_level
                    .increase_indent(make::if_expression(&cond_expr, early_expression));
                replace(new_expr.syntax(), &then_block, &parent_block, &if_expr)
            }
            Some((path, bound_ident)) => {
                // If-let.
                let match_expr = {
                    let happy_arm = make::match_arm(
                        once(
                            make::tuple_struct_pat(
                                path,
                                once(make::bind_pat(make::name("it")).into()),
                            )
                            .into(),
                        ),
                        make::expr_path(make::path_from_name_ref(make::name_ref("it"))),
                    );

                    let sad_arm = make::match_arm(
                        // FIXME: would be cool to use `None` or `Err(_)` if appropriate
                        once(make::placeholder_pat().into()),
                        early_expression,
                    );

                    make::expr_match(cond_expr, make::match_arm_list(vec![happy_arm, sad_arm]))
                };

                let let_stmt = make::let_stmt(
                    make::bind_pat(make::name(&bound_ident.syntax().to_string())).into(),
                    Some(match_expr),
                );
                let let_stmt = if_indent_level.increase_indent(let_stmt);
                replace(let_stmt.syntax(), &then_block, &parent_block, &if_expr)
            }
        };
        edit.target(if_expr.syntax().text_range());
        edit.replace_ast(parent_block, ast::Block::cast(new_block).unwrap());
        edit.set_cursor(cursor_position);

        fn replace(
            new_expr: &SyntaxNode,
            then_block: &Block,
            parent_block: &Block,
            if_expr: &ast::IfExpr,
        ) -> SyntaxNode {
            let then_block_items = IndentLevel::from(1).decrease_indent(then_block.clone());
            let end_of_then = then_block_items.syntax().last_child_or_token().unwrap();
            let end_of_then =
                if end_of_then.prev_sibling_or_token().map(|n| n.kind()) == Some(WHITESPACE) {
                    end_of_then.prev_sibling_or_token().unwrap()
                } else {
                    end_of_then
                };
            let mut then_statements = new_expr.children_with_tokens().chain(
                then_block_items
                    .syntax()
                    .children_with_tokens()
                    .skip(1)
                    .take_while(|i| *i != end_of_then),
            );
            replace_children(
                &parent_block.syntax(),
                RangeInclusive::new(
                    if_expr.clone().syntax().clone().into(),
                    if_expr.syntax().clone().into(),
                ),
                &mut then_statements,
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn convert_inside_fn() {
        check_assist(
            convert_to_guarded_return,
            r#"
            fn main() {
                bar();
                if<|> true {
                    foo();

                    //comment
                    bar();
                }
            }
            "#,
            r#"
            fn main() {
                bar();
                if<|> !true {
                    return;
                }
                foo();

                //comment
                bar();
            }
            "#,
        );
    }

    #[test]
    fn convert_let_inside_fn() {
        check_assist(
            convert_to_guarded_return,
            r#"
            fn main(n: Option<String>) {
                bar();
                if<|> let Some(n) = n {
                    foo(n);

                    //comment
                    bar();
                }
            }
            "#,
            r#"
            fn main(n: Option<String>) {
                bar();
                le<|>t n = match n {
                    Some(it) => it,
                    _ => return,
                };
                foo(n);

                //comment
                bar();
            }
            "#,
        );
    }

    #[test]
    fn convert_if_let_result() {
        check_assist(
            convert_to_guarded_return,
            r#"
            fn main() {
                if<|> let Ok(x) = Err(92) {
                    foo(x);
                }
            }
            "#,
            r#"
            fn main() {
                le<|>t x = match Err(92) {
                    Ok(it) => it,
                    _ => return,
                };
                foo(x);
            }
            "#,
        );
    }

    #[test]
    fn convert_let_ok_inside_fn() {
        check_assist(
            convert_to_guarded_return,
            r#"
            fn main(n: Option<String>) {
                bar();
                if<|> let Ok(n) = n {
                    foo(n);

                    //comment
                    bar();
                }
            }
            "#,
            r#"
            fn main(n: Option<String>) {
                bar();
                le<|>t n = match n {
                    Ok(it) => it,
                    _ => return,
                };
                foo(n);

                //comment
                bar();
            }
            "#,
        );
    }

    #[test]
    fn convert_inside_while() {
        check_assist(
            convert_to_guarded_return,
            r#"
            fn main() {
                while true {
                    if<|> true {
                        foo();
                        bar();
                    }
                }
            }
            "#,
            r#"
            fn main() {
                while true {
                    if<|> !true {
                        continue;
                    }
                    foo();
                    bar();
                }
            }
            "#,
        );
    }

    #[test]
    fn convert_let_inside_while() {
        check_assist(
            convert_to_guarded_return,
            r#"
            fn main() {
                while true {
                    if<|> let Some(n) = n {
                        foo(n);
                        bar();
                    }
                }
            }
            "#,
            r#"
            fn main() {
                while true {
                    le<|>t n = match n {
                        Some(it) => it,
                        _ => continue,
                    };
                    foo(n);
                    bar();
                }
            }
            "#,
        );
    }

    #[test]
    fn convert_inside_loop() {
        check_assist(
            convert_to_guarded_return,
            r#"
            fn main() {
                loop {
                    if<|> true {
                        foo();
                        bar();
                    }
                }
            }
            "#,
            r#"
            fn main() {
                loop {
                    if<|> !true {
                        continue;
                    }
                    foo();
                    bar();
                }
            }
            "#,
        );
    }

    #[test]
    fn convert_let_inside_loop() {
        check_assist(
            convert_to_guarded_return,
            r#"
            fn main() {
                loop {
                    if<|> let Some(n) = n {
                        foo(n);
                        bar();
                    }
                }
            }
            "#,
            r#"
            fn main() {
                loop {
                    le<|>t n = match n {
                        Some(it) => it,
                        _ => continue,
                    };
                    foo(n);
                    bar();
                }
            }
            "#,
        );
    }

    #[test]
    fn ignore_already_converted_if() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
            fn main() {
                if<|> true {
                    return;
                }
            }
            "#,
        );
    }

    #[test]
    fn ignore_already_converted_loop() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
            fn main() {
                loop {
                    if<|> true {
                        continue;
                    }
                }
            }
            "#,
        );
    }

    #[test]
    fn ignore_return() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
            fn main() {
                if<|> true {
                    return
                }
            }
            "#,
        );
    }

    #[test]
    fn ignore_else_branch() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
            fn main() {
                if<|> true {
                    foo();
                } else {
                    bar()
                }
            }
            "#,
        );
    }

    #[test]
    fn ignore_statements_aftert_if() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
            fn main() {
                if<|> true {
                    foo();
                }
                bar();
            }
            "#,
        );
    }

    #[test]
    fn ignore_statements_inside_if() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
            fn main() {
                if false {
                    if<|> true {
                        foo();
                    }
                }
            }
            "#,
        );
    }
}
