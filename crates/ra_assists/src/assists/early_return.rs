use std::ops::RangeInclusive;

use hir::db::HirDatabase;
use ra_syntax::{
    algo::replace_children,
    ast::{self, edit::IndentLevel, make},
    AstNode,
    SyntaxKind::{FN_DEF, LOOP_EXPR, L_CURLY, R_CURLY, WHILE_EXPR, WHITESPACE},
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
pub(crate) fn convert_to_guarded_return(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let if_expr: ast::IfExpr = ctx.find_node_at_offset()?;
    let expr = if_expr.condition()?.expr()?;
    let then_block = if_expr.then_branch()?.block()?;
    if if_expr.else_branch().is_some() {
        return None;
    }

    let parent_block = if_expr.syntax().parent()?.ancestors().find_map(ast::Block::cast)?;

    if parent_block.expr()? != if_expr.clone().into() {
        return None;
    }

    // check for early return and continue
    let first_in_then_block = then_block.syntax().first_child()?.clone();
    if ast::ReturnExpr::can_cast(first_in_then_block.kind())
        || ast::ContinueExpr::can_cast(first_in_then_block.kind())
        || first_in_then_block
            .children()
            .any(|x| ast::ReturnExpr::can_cast(x.kind()) || ast::ContinueExpr::can_cast(x.kind()))
    {
        return None;
    }

    let parent_container = parent_block.syntax().parent()?.parent()?;

    let early_expression = match parent_container.kind() {
        WHILE_EXPR | LOOP_EXPR => Some("continue;"),
        FN_DEF => Some("return;"),
        _ => None,
    }?;

    if then_block.syntax().first_child_or_token().map(|t| t.kind() == L_CURLY).is_none() {
        return None;
    }

    then_block.syntax().last_child_or_token().filter(|t| t.kind() == R_CURLY)?;
    let cursor_position = ctx.frange.range.start();

    ctx.add_action(AssistId("convert_to_guarded_return"), "convert to guarded return", |edit| {
        let if_indent_level = IndentLevel::from_node(&if_expr.syntax());
        let new_if_expr =
            if_indent_level.increase_indent(make::if_expression(&expr, early_expression));
        let then_block_items = IndentLevel::from(1).decrease_indent(then_block.clone());
        let end_of_then = then_block_items.syntax().last_child_or_token().unwrap();
        let end_of_then =
            if end_of_then.prev_sibling_or_token().map(|n| n.kind()) == Some(WHITESPACE) {
                end_of_then.prev_sibling_or_token().unwrap()
            } else {
                end_of_then
            };
        let mut new_if_and_then_statements = new_if_expr.syntax().children_with_tokens().chain(
            then_block_items
                .syntax()
                .children_with_tokens()
                .skip(1)
                .take_while(|i| *i != end_of_then),
        );
        let new_block = replace_children(
            &parent_block.syntax(),
            RangeInclusive::new(
                if_expr.clone().syntax().clone().into(),
                if_expr.syntax().clone().into(),
            ),
            &mut new_if_and_then_statements,
        );
        edit.target(if_expr.syntax().text_range());
        edit.replace_ast(parent_block, ast::Block::cast(new_block).unwrap());
        edit.set_cursor(cursor_position);
    });
    ctx.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_not_applicable};

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
