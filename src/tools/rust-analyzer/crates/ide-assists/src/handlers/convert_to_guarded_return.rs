use std::iter::once;

use ide_db::{
    syntax_helpers::node_ext::{is_pattern_cond, single_let},
    ty_filter::TryEnum,
};
use syntax::{
    AstNode,
    SyntaxKind::{FN, FOR_EXPR, LOOP_EXPR, WHILE_EXPR, WHITESPACE},
    T,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
    utils::invert_boolean_expression_legacy,
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
    if let Some(let_stmt) = ctx.find_node_at_offset() {
        let_stmt_to_guarded_return(let_stmt, acc, ctx)
    } else if let Some(if_expr) = ctx.find_node_at_offset() {
        if_expr_to_guarded_return(if_expr, acc, ctx)
    } else {
        None
    }
}

fn if_expr_to_guarded_return(
    if_expr: ast::IfExpr,
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    if if_expr.else_branch().is_some() {
        return None;
    }

    let cond = if_expr.condition()?;

    let if_token_range = if_expr.if_token()?.text_range();
    let if_cond_range = cond.syntax().text_range();

    let cursor_in_range =
        if_token_range.cover(if_cond_range).contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    // Check if there is an IfLet that we can handle.
    let (if_let_pat, cond_expr) = if is_pattern_cond(cond.clone()) {
        let let_ = single_let(cond)?;
        (Some(let_.pat()?), let_.expr()?)
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
        WHILE_EXPR | LOOP_EXPR | FOR_EXPR => make::expr_continue(None),
        FN => make::expr_return(None),
        _ => return None,
    };

    then_block.syntax().first_child_or_token().map(|t| t.kind() == T!['{'])?;

    then_block.syntax().last_child_or_token().filter(|t| t.kind() == T!['}'])?;

    let then_block_items = then_block.dedent(IndentLevel(1));

    let end_of_then = then_block_items.syntax().last_child_or_token()?;
    let end_of_then = if end_of_then.prev_sibling_or_token().map(|n| n.kind()) == Some(WHITESPACE) {
        end_of_then.prev_sibling_or_token()?
    } else {
        end_of_then
    };

    let target = if_expr.syntax().text_range();
    acc.add(
        AssistId::refactor_rewrite("convert_to_guarded_return"),
        "Convert to guarded return",
        target,
        |edit| {
            let if_indent_level = IndentLevel::from_node(if_expr.syntax());
            let replacement = match if_let_pat {
                None => {
                    // If.
                    let new_expr = {
                        let then_branch =
                            make::block_expr(once(make::expr_stmt(early_expression).into()), None);
                        let cond = invert_boolean_expression_legacy(cond_expr);
                        make::expr_if(cond, then_branch, None).indent(if_indent_level)
                    };
                    new_expr.syntax().clone()
                }
                Some(pat) => {
                    // If-let.
                    let let_else_stmt = make::let_else_stmt(
                        pat,
                        None,
                        cond_expr,
                        ast::make::tail_only_block_expr(early_expression),
                    );
                    let let_else_stmt = let_else_stmt.indent(if_indent_level);
                    let_else_stmt.syntax().clone()
                }
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
            let mut editor = edit.make_editor(if_expr.syntax());
            editor.replace_with_many(if_expr.syntax(), then_statements);
            edit.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn let_stmt_to_guarded_return(
    let_stmt: ast::LetStmt,
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let pat = let_stmt.pat()?;
    let expr = let_stmt.initializer()?;

    let let_token_range = let_stmt.let_token()?.text_range();
    let let_pattern_range = pat.syntax().text_range();
    let cursor_in_range =
        let_token_range.cover(let_pattern_range).contains_range(ctx.selection_trimmed());

    if !cursor_in_range {
        return None;
    }

    let try_enum =
        ctx.sema.type_of_expr(&expr).and_then(|ty| TryEnum::from_ty(&ctx.sema, &ty.adjusted()))?;

    let happy_pattern = try_enum.happy_pattern(pat);
    let target = let_stmt.syntax().text_range();

    let early_expression: ast::Expr = {
        let parent_block =
            let_stmt.syntax().parent()?.ancestors().find_map(ast::BlockExpr::cast)?;
        let parent_container = parent_block.syntax().parent()?;

        match parent_container.kind() {
            WHILE_EXPR | LOOP_EXPR | FOR_EXPR => make::expr_continue(None),
            FN => make::expr_return(None),
            _ => return None,
        }
    };

    acc.add(
        AssistId::refactor_rewrite("convert_to_guarded_return"),
        "Convert to guarded return",
        target,
        |edit| {
            let let_indent_level = IndentLevel::from_node(let_stmt.syntax());

            let replacement = {
                let let_else_stmt = make::let_else_stmt(
                    happy_pattern,
                    let_stmt.ty(),
                    expr,
                    ast::make::tail_only_block_expr(early_expression),
                );
                let let_else_stmt = let_else_stmt.indent(let_indent_level);
                let_else_stmt.syntax().clone()
            };
            let mut editor = edit.make_editor(let_stmt.syntax());
            editor.replace(let_stmt.syntax(), replacement);
            edit.add_file_edits(ctx.vfs_file_id(), editor);
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
    fn convert_let_inside_for() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    for n in ns {
        if$0 let Some(n) = n {
            foo(n);
            bar();
        }
    }
}
"#,
            r#"
fn main() {
    for n in ns {
        let Some(n) = n else { continue };
        foo(n);
        bar();
    }
}
"#,
        );
    }

    #[test]
    fn convert_let_stmt_inside_fn() {
        check_assist(
            convert_to_guarded_return,
            r#"
//- minicore: option
fn foo() -> Option<i32> {
    None
}

fn main() {
    let x$0 = foo();
}
"#,
            r#"
fn foo() -> Option<i32> {
    None
}

fn main() {
    let Some(x) = foo() else { return };
}
"#,
        );
    }

    #[test]
    fn convert_let_stmt_inside_loop() {
        check_assist(
            convert_to_guarded_return,
            r#"
//- minicore: option
fn foo() -> Option<i32> {
    None
}

fn main() {
    loop {
        let x$0 = foo();
    }
}
"#,
            r#"
fn foo() -> Option<i32> {
    None
}

fn main() {
    loop {
        let Some(x) = foo() else { continue };
    }
}
"#,
        );
    }

    #[test]
    fn convert_arbitrary_if_let_patterns() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    $0if let None = Some(92) {
        foo();
    }
}
"#,
            r#"
fn main() {
    let None = Some(92) else { return };
    foo();
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    $0if let [1, x] = [1, 92] {
        foo(x);
    }
}
"#,
            r#"
fn main() {
    let [1, x] = [1, 92] else { return };
    foo(x);
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    $0if let (Some(x), None) = (Some(92), None) {
        foo(x);
    }
}
"#,
            r#"
fn main() {
    let (Some(x), None) = (Some(92), None) else { return };
    foo(x);
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

    #[test]
    fn ignore_inside_if_stmt() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
fn main() {
    if false {
        foo()$0;
    }
}
"#,
        );
    }

    #[test]
    fn ignore_inside_let_initializer() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
//- minicore: option
fn foo() -> Option<i32> {
    None
}

fn main() {
    let x = foo()$0;
}
"#,
        );
    }
}
