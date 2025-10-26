use std::iter::once;

use either::Either;
use hir::{Semantics, TypeInfo};
use ide_db::{RootDatabase, ty_filter::TryEnum};
use syntax::{
    AstNode,
    SyntaxKind::{CLOSURE_EXPR, FN, FOR_EXPR, LOOP_EXPR, WHILE_EXPR, WHITESPACE},
    SyntaxNode, T,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
    utils::{invert_boolean_expression_legacy, is_never_block},
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
    match ctx.find_node_at_offset::<Either<ast::LetStmt, ast::IfExpr>>()? {
        Either::Left(let_stmt) => let_stmt_to_guarded_return(let_stmt, acc, ctx),
        Either::Right(if_expr) => if_expr_to_guarded_return(if_expr, acc, ctx),
    }
}

fn if_expr_to_guarded_return(
    if_expr: ast::IfExpr,
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let else_block = match if_expr.else_branch() {
        Some(ast::ElseBranch::Block(block_expr)) if is_never_block(&ctx.sema, &block_expr) => {
            Some(block_expr)
        }
        Some(_) => return None,
        _ => None,
    };

    let cond = if_expr.condition()?;

    let if_token_range = if_expr.if_token()?.text_range();
    let if_cond_range = cond.syntax().text_range();

    let cursor_in_range =
        if_token_range.cover(if_cond_range).contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    let let_chains = flat_let_chain(cond);

    let then_branch = if_expr.then_branch()?;
    let then_block = then_branch.stmt_list()?;

    let parent_block = if_expr.syntax().parent()?.ancestors().find_map(ast::BlockExpr::cast)?;

    if parent_block.tail_expr()? != if_expr.clone().into() {
        return None;
    }

    // check for early return and continue
    if is_early_block(&then_block) || is_never_block(&ctx.sema, &then_branch) {
        return None;
    }

    let parent_container = parent_block.syntax().parent()?;

    let early_expression = else_block
        .or_else(|| {
            early_expression(parent_container, &ctx.sema).map(ast::make::tail_only_block_expr)
        })?
        .reset_indent();

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
            let replacement = let_chains.into_iter().map(|expr| {
                if let ast::Expr::LetExpr(let_expr) = &expr
                    && let (Some(pat), Some(expr)) = (let_expr.pat(), let_expr.expr())
                {
                    // If-let.
                    let let_else_stmt =
                        make::let_else_stmt(pat, None, expr, early_expression.clone());
                    let let_else_stmt = let_else_stmt.indent(if_indent_level);
                    let_else_stmt.syntax().clone()
                } else {
                    // If.
                    let new_expr = {
                        let then_branch = clean_stmt_block(&early_expression);
                        let cond = invert_boolean_expression_legacy(expr);
                        make::expr_if(cond, then_branch, None).indent(if_indent_level)
                    };
                    new_expr.syntax().clone()
                }
            });

            let newline = &format!("\n{if_indent_level}");
            let then_statements = replacement
                .enumerate()
                .flat_map(|(i, node)| {
                    (i != 0)
                        .then(|| make::tokens::whitespace(newline).into())
                        .into_iter()
                        .chain(node.children_with_tokens())
                })
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

    if !cursor_in_range || let_stmt.let_else().is_some() {
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

        early_expression(parent_container, &ctx.sema)?
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

fn early_expression(
    parent_container: SyntaxNode,
    sema: &Semantics<'_, RootDatabase>,
) -> Option<ast::Expr> {
    let return_none_expr = || {
        let none_expr = make::expr_path(make::ext::ident_path("None"));
        make::expr_return(Some(none_expr))
    };
    if let Some(fn_) = ast::Fn::cast(parent_container.clone())
        && let Some(fn_def) = sema.to_def(&fn_)
        && let Some(TryEnum::Option) = TryEnum::from_ty(sema, &fn_def.ret_type(sema.db))
    {
        return Some(return_none_expr());
    }
    if let Some(body) = ast::ClosureExpr::cast(parent_container.clone()).and_then(|it| it.body())
        && let Some(ret_ty) = sema.type_of_expr(&body).map(TypeInfo::original)
        && let Some(TryEnum::Option) = TryEnum::from_ty(sema, &ret_ty)
    {
        return Some(return_none_expr());
    }

    Some(match parent_container.kind() {
        WHILE_EXPR | LOOP_EXPR | FOR_EXPR => make::expr_continue(None),
        FN | CLOSURE_EXPR => make::expr_return(None),
        _ => return None,
    })
}

fn flat_let_chain(mut expr: ast::Expr) -> Vec<ast::Expr> {
    let mut chains = vec![];
    let mut reduce_cond = |rhs| {
        if !matches!(rhs, ast::Expr::LetExpr(_))
            && let Some(last) = chains.pop_if(|last| !matches!(last, ast::Expr::LetExpr(_)))
        {
            chains.push(make::expr_bin_op(rhs, ast::BinaryOp::LogicOp(ast::LogicOp::And), last));
        } else {
            chains.push(rhs);
        }
    };

    while let ast::Expr::BinExpr(bin_expr) = &expr
        && bin_expr.op_kind() == Some(ast::BinaryOp::LogicOp(ast::LogicOp::And))
        && let (Some(lhs), Some(rhs)) = (bin_expr.lhs(), bin_expr.rhs())
    {
        reduce_cond(rhs);
        expr = lhs;
    }

    reduce_cond(expr);
    chains.reverse();
    chains
}

fn clean_stmt_block(block: &ast::BlockExpr) -> ast::BlockExpr {
    if block.statements().next().is_none()
        && let Some(tail_expr) = block.tail_expr()
        && block.modifier().is_none()
    {
        make::block_expr(once(make::expr_stmt(tail_expr).into()), None)
    } else {
        block.clone()
    }
}

fn is_early_block(then_block: &ast::StmtList) -> bool {
    let is_early_expr =
        |expr| matches!(expr, ast::Expr::ReturnExpr(_) | ast::Expr::ContinueExpr(_));
    let into_expr = |stmt| match stmt {
        ast::Stmt::ExprStmt(expr_stmt) => expr_stmt.expr(),
        _ => None,
    };
    then_block.tail_expr().is_some_and(is_early_expr)
        || then_block.statements().filter_map(into_expr).any(is_early_expr)
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
    fn convert_inside_fn_return_option() {
        check_assist(
            convert_to_guarded_return,
            r#"
//- minicore: option
fn ret_option() -> Option<()> {
    bar();
    if$0 true {
        foo();

        // comment
        bar();
    }
}
"#,
            r#"
fn ret_option() -> Option<()> {
    bar();
    if false {
        return None;
    }
    foo();

    // comment
    bar();
}
"#,
        );
    }

    #[test]
    fn convert_inside_closure() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    let _f = || {
        bar();
        if$0 true {
            foo();

            // comment
            bar();
        }
    }
}
"#,
            r#"
fn main() {
    let _f = || {
        bar();
        if false {
            return;
        }
        foo();

        // comment
        bar();
    }
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
    fn convert_if_let_has_never_type_else_block() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 let Ok(x) = Err(92) {
        foo(x);
    } else {
        // needless comment
        return;
    }
}
"#,
            r#"
fn main() {
    let Ok(x) = Err(92) else {
        // needless comment
        return;
    };
    foo(x);
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 let Ok(x) = Err(92) {
        foo(x);
    } else {
        return
    }
}
"#,
            r#"
fn main() {
    let Ok(x) = Err(92) else {
        return
    };
    foo(x);
}
"#,
        );
    }

    #[test]
    fn convert_if_let_result_inside_let() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    let _x = loop {
        if$0 let Ok(x) = Err(92) {
            foo(x);
        }
    };
}
"#,
            r#"
fn main() {
    let _x = loop {
        let Ok(x) = Err(92) else { continue };
        foo(x);
    };
}
"#,
        );
    }

    #[test]
    fn convert_if_let_chain_result() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 let Ok(x) = Err(92)
        && x < 30
        && let Some(y) = Some(8)
    {
        foo(x, y);
    }
}
"#,
            r#"
fn main() {
    let Ok(x) = Err(92) else { return };
    if x >= 30 {
        return;
    }
    let Some(y) = Some(8) else { return };
    foo(x, y);
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 let Ok(x) = Err(92)
        && x < 30
        && y < 20
        && let Some(y) = Some(8)
    {
        foo(x, y);
    }
}
"#,
            r#"
fn main() {
    let Ok(x) = Err(92) else { return };
    if !(x < 30 && y < 20) {
        return;
    }
    let Some(y) = Some(8) else { return };
    foo(x, y);
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 let Ok(x) = Err(92)
        && let Ok(y) = Ok(37)
        && x < 30
        && let Some(y) = Some(8)
    {
        foo(x, y);
    }
}
"#,
            r#"
fn main() {
    let Ok(x) = Err(92) else { return };
    let Ok(y) = Ok(37) else { return };
    if x >= 30 {
        return;
    }
    let Some(y) = Some(8) else { return };
    foo(x, y);
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 cond
        && let Ok(x) = Err(92)
        && let Ok(y) = Ok(37)
        && x < 30
        && let Some(y) = Some(8)
    {
        foo(x, y);
    }
}
"#,
            r#"
fn main() {
    if !cond {
        return;
    }
    let Ok(x) = Err(92) else { return };
    let Ok(y) = Ok(37) else { return };
    if x >= 30 {
        return;
    }
    let Some(y) = Some(8) else { return };
    foo(x, y);
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 cond
        && foo()
        && let Ok(x) = Err(92)
        && let Ok(y) = Ok(37)
        && x < 30
        && let Some(y) = Some(8)
    {
        foo(x, y);
    }
}
"#,
            r#"
fn main() {
    if !(cond && foo()) {
        return;
    }
    let Ok(x) = Err(92) else { return };
    let Ok(y) = Ok(37) else { return };
    if x >= 30 {
        return;
    }
    let Some(y) = Some(8) else { return };
    foo(x, y);
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
    fn convert_let_stmt_inside_fn_return_option() {
        check_assist(
            convert_to_guarded_return,
            r#"
//- minicore: option
fn foo() -> Option<i32> {
    None
}

fn ret_option() -> Option<i32> {
    let x$0 = foo();
}
"#,
            r#"
fn foo() -> Option<i32> {
    None
}

fn ret_option() -> Option<i32> {
    let Some(x) = foo() else { return None };
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
    fn ignore_let_else_branch() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
//- minicore: option
fn main() {
    let$0 Some(x) = Some(2) else { return };
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
