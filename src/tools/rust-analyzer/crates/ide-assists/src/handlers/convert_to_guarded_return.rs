use std::iter::once;

use ide_db::syntax_helpers::node_ext::{is_pattern_cond, single_let};
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    ted, AstNode,
    SyntaxKind::{FN, LOOP_EXPR, WHILE_EXPR, WHITESPACE},
    T,
};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::invert_boolean_expression,
    AssistId, AssistKind,
};

// Assist: convert_to_guarded_return
//
// Replace a large conditional with a guarded return.
//
// ```
// fn main() {
//     $0if cond {
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
pub(crate) fn convert_to_guarded_return(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let if_expr: ast::IfExpr = ctx.find_node_at_offset()?;
    if if_expr.else_branch().is_some() {
        return None;
    }

    let cond = if_expr.condition()?;

    // Check if there is an IfLet that we can handle.
    let (if_let_pat, cond_expr) = if is_pattern_cond(cond.clone()) {
        let let_ = single_let(cond)?;
        match let_.pat() {
            Some(ast::Pat::TupleStructPat(pat)) if pat.fields().count() == 1 => {
                let path = pat.path()?;
                if path.qualifier().is_some() {
                    return None;
                }

                let bound_ident = pat.fields().next().unwrap();
                if !ast::IdentPat::can_cast(bound_ident.syntax().kind()) {
                    return None;
                }

                (Some((path, bound_ident)), let_.expr()?)
            }
            _ => return None, // Unsupported IfLet.
        }
    } else {
        (None, cond)
    };

    let then_block = if_expr.then_branch()?;
    let then_block = then_block.stmt_list()?;

    let parent_block = if_expr.syntax().parent()?.ancestors().find_map(ast::BlockExpr::cast)?;

    if parent_block.tail_expr()? != if_expr.clone().into() {
        return None;
    }

    // FIXME: This relies on untyped syntax tree and casts to much. It should be
    // rewritten to use strongly-typed APIs.

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

    let parent_container = parent_block.syntax().parent()?;

    let early_expression: ast::Expr = match parent_container.kind() {
        WHILE_EXPR | LOOP_EXPR => make::expr_continue(None),
        FN => make::expr_return(None),
        _ => return None,
    };

    if then_block.syntax().first_child_or_token().map(|t| t.kind() == T!['{']).is_none() {
        return None;
    }

    then_block.syntax().last_child_or_token().filter(|t| t.kind() == T!['}'])?;

    let target = if_expr.syntax().text_range();
    acc.add(
        AssistId("convert_to_guarded_return", AssistKind::RefactorRewrite),
        "Convert to guarded return",
        target,
        |edit| {
            let if_expr = edit.make_mut(if_expr);
            let if_indent_level = IndentLevel::from_node(if_expr.syntax());
            let replacement = match if_let_pat {
                None => {
                    // If.
                    let new_expr = {
                        let then_branch =
                            make::block_expr(once(make::expr_stmt(early_expression).into()), None);
                        let cond = invert_boolean_expression(cond_expr);
                        make::expr_if(cond, then_branch, None).indent(if_indent_level)
                    };
                    new_expr.syntax().clone_for_update()
                }
                Some((path, bound_ident)) => {
                    // If-let.
                    let pat = make::tuple_struct_pat(path, once(bound_ident));
                    let let_else_stmt = make::let_else_stmt(
                        pat.into(),
                        None,
                        cond_expr,
                        ast::make::tail_only_block_expr(early_expression),
                    );
                    let let_else_stmt = let_else_stmt.indent(if_indent_level);
                    let_else_stmt.syntax().clone_for_update()
                }
            };

            let then_block_items = then_block.dedent(IndentLevel(1)).clone_for_update();

            let end_of_then = then_block_items.syntax().last_child_or_token().unwrap();
            let end_of_then =
                if end_of_then.prev_sibling_or_token().map(|n| n.kind()) == Some(WHITESPACE) {
                    end_of_then.prev_sibling_or_token().unwrap()
                } else {
                    end_of_then
                };

            let then_statements = replacement
                .children_with_tokens()
                .chain(
                    then_block_items
                        .syntax()
                        .children_with_tokens()
                        .skip(1)
                        .take_while(|i| *i != end_of_then),
                )
                .collect();

            ted::replace_with_many(if_expr.syntax(), then_statements)
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
            convert_to_guarded_return,
            r#"
fn main() {
    bar();
    if$0 true {
        foo();

        // comment
        bar();
    }
}
"#,
            r#"
fn main() {
    bar();
    if false {
        return;
    }
    foo();

    // comment
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
    if$0 let Some(n) = n {
        foo(n);

        // comment
        bar();
    }
}
"#,
            r#"
fn main(n: Option<String>) {
    bar();
    let Some(n) = n else { return };
    foo(n);

    // comment
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
    if$0 let Ok(x) = Err(92) {
        foo(x);
    }
}
"#,
            r#"
fn main() {
    let Ok(x) = Err(92) else { return };
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
    if$0 let Some(n) = n {
        foo(n);

        // comment
        bar();
    }
}
"#,
            r#"
fn main(n: Option<String>) {
    bar();
    let Some(n) = n else { return };
    foo(n);

    // comment
    bar();
}
"#,
        );
    }

    #[test]
    fn convert_let_mut_ok_inside_fn() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main(n: Option<String>) {
    bar();
    if$0 let Some(mut n) = n {
        foo(n);

        // comment
        bar();
    }
}
"#,
            r#"
fn main(n: Option<String>) {
    bar();
    let Some(mut n) = n else { return };
    foo(n);

    // comment
    bar();
}
"#,
        );
    }

    #[test]
    fn convert_let_ref_ok_inside_fn() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main(n: Option<&str>) {
    bar();
    if$0 let Some(ref n) = n {
        foo(n);

        // comment
        bar();
    }
}
"#,
            r#"
fn main(n: Option<&str>) {
    bar();
    let Some(ref n) = n else { return };
    foo(n);

    // comment
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
        if$0 true {
            foo();
            bar();
        }
    }
}
"#,
            r#"
fn main() {
    while true {
        if false {
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
        if$0 let Some(n) = n {
            foo(n);
            bar();
        }
    }
}
"#,
            r#"
fn main() {
    while true {
        let Some(n) = n else { continue };
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
        if$0 true {
            foo();
            bar();
        }
    }
}
"#,
            r#"
fn main() {
    loop {
        if false {
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
        if$0 let Some(n) = n {
            foo(n);
            bar();
        }
    }
}
"#,
            r#"
fn main() {
    loop {
        let Some(n) = n else { continue };
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
    if$0 true {
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
        if$0 true {
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
    if$0 true {
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
    if$0 true {
        foo();
    } else {
        bar()
    }
}
"#,
        );
    }

    #[test]
    fn ignore_statements_after_if() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 true {
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
        if$0 true {
            foo();
        }
    }
}
"#,
        );
    }
}
