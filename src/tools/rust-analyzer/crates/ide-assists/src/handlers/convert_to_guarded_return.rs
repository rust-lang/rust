use std::iter::once;

use either::Either;
use hir::Semantics;
use ide_db::{RootDatabase, ty_filter::TryEnum};
use syntax::{
    AstNode,
    SyntaxKind::WHITESPACE,
    SyntaxNode, T,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        syntax_factory::SyntaxFactory,
    },
    match_ast,
    syntax_editor::SyntaxEditor,
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
    utils::{invert_boolean_expression, is_never_block},
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
// ---
// ```
// //- minicore: option
// fn foo() -> Option<i32> { None }
// fn main() {
//     $0let x = foo();
// }
// ```
// ->
// ```
// fn foo() -> Option<i32> { None }
// fn main() {
//     let Some(x) = foo() else { return };
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
    let make = SyntaxFactory::without_mappings();
    let else_block = match if_expr.else_branch() {
        Some(ast::ElseBranch::Block(block_expr)) => Some(block_expr),
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

    let let_chains = flat_let_chain(cond, &make);

    let then_branch = if_expr.then_branch()?;
    let then_block = then_branch.stmt_list()?;

    let parent_block = if_expr.syntax().parent()?.ancestors().find_map(ast::BlockExpr::cast)?;

    // check for early return and continue
    if is_early_block(&then_block) || is_never_block(&ctx.sema, &then_branch) {
        return None;
    }

    let parent_container = parent_block.syntax().parent()?;
    let else_block = ElseBlock::new(&ctx.sema, else_block, &parent_container)?;

    if parent_block.tail_expr() != Some(if_expr.clone().into())
        && !(else_block.is_never_block
            && ast::ExprStmt::can_cast(if_expr.syntax().parent()?.kind()))
    {
        return None;
    }

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
            let make = SyntaxFactory::without_mappings();
            let if_indent_level = IndentLevel::from_node(if_expr.syntax());
            let early_expression = else_block.make_early_block(&ctx.sema, &make);
            let replacement = let_chains.into_iter().map(|expr| {
                if let ast::Expr::LetExpr(let_expr) = &expr
                    && let (Some(pat), Some(expr)) = (let_expr.pat(), let_expr.expr())
                {
                    // If-let.
                    let let_else_stmt =
                        make.let_else_stmt(pat, None, expr, early_expression.clone());
                    let let_else_stmt = let_else_stmt.indent(if_indent_level);
                    let_else_stmt.syntax().clone()
                } else {
                    // If.
                    let new_expr = {
                        let then_branch = clean_stmt_block(&early_expression, &make);
                        let cond = invert_boolean_expression(&make, expr);
                        make.expr_if(cond, then_branch, None).indent(if_indent_level)
                    };
                    new_expr.syntax().clone()
                }
            });

            let newline = &format!("\n{if_indent_level}");
            let then_statements = replacement
                .enumerate()
                .flat_map(|(i, node)| {
                    (i != 0)
                        .then(|| make.whitespace(newline).into())
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

    let parent_block = let_stmt.syntax().parent()?.ancestors().find_map(ast::BlockExpr::cast)?;
    let parent_container = parent_block.syntax().parent()?;
    let else_block = ElseBlock::new(&ctx.sema, None, &parent_container)?;

    acc.add(
        AssistId::refactor_rewrite("convert_to_guarded_return"),
        "Convert to guarded return",
        target,
        |edit| {
            let let_indent_level = IndentLevel::from_node(let_stmt.syntax());
            let make = SyntaxFactory::without_mappings();

            let replacement = {
                let let_else_stmt = make.let_else_stmt(
                    happy_pattern,
                    let_stmt.ty(),
                    expr.reset_indent(),
                    else_block.make_early_block(&ctx.sema, &make),
                );
                let let_else_stmt = let_else_stmt.indent(let_indent_level);
                let_else_stmt.syntax().clone()
            };
            let mut editor = edit.make_editor(let_stmt.syntax());
            editor.replace(let_stmt.syntax(), replacement);
            editor.add_mappings(make.finish_with_mappings());
            edit.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

struct ElseBlock<'db> {
    exist_else_block: Option<ast::BlockExpr>,
    is_never_block: bool,
    kind: EarlyKind<'db>,
}

impl<'db> ElseBlock<'db> {
    fn new(
        sema: &Semantics<'db, RootDatabase>,
        exist_else_block: Option<ast::BlockExpr>,
        parent_container: &SyntaxNode,
    ) -> Option<Self> {
        let is_never_block = exist_else_block.as_ref().is_some_and(|it| is_never_block(sema, it));
        let kind = EarlyKind::from_node(parent_container, sema)?;

        Some(Self { exist_else_block, is_never_block, kind })
    }

    fn make_early_block(
        self,
        sema: &Semantics<'_, RootDatabase>,
        make: &SyntaxFactory,
    ) -> ast::BlockExpr {
        let Some(block_expr) = self.exist_else_block else {
            return make.tail_only_block_expr(self.kind.make_early_expr(sema, make, None));
        };

        if self.is_never_block {
            return block_expr.reset_indent();
        }

        let (mut edit, block_expr) = SyntaxEditor::with_ast_node(&block_expr.reset_indent());

        let last_stmt = block_expr.statements().last().map(|it| it.syntax().clone());
        let tail_expr = block_expr.tail_expr().map(|it| it.syntax().clone());
        let Some(last_element) = tail_expr.clone().or(last_stmt.clone()) else {
            return make.tail_only_block_expr(self.kind.make_early_expr(sema, make, None));
        };
        let whitespace = last_element.prev_sibling_or_token().filter(|it| it.kind() == WHITESPACE);

        let make = SyntaxFactory::without_mappings();

        if let Some(tail_expr) = block_expr.tail_expr()
            && !self.kind.is_unit()
        {
            let early_expr = self.kind.make_early_expr(sema, &make, Some(tail_expr.clone()));
            edit.replace(tail_expr.syntax(), early_expr.syntax());
        } else {
            let last_stmt = match block_expr.tail_expr() {
                Some(expr) => make.expr_stmt(expr).syntax().clone(),
                None => last_element.clone(),
            };
            let whitespace =
                make.whitespace(&whitespace.map_or(String::new(), |it| it.to_string()));
            let early_expr = self.kind.make_early_expr(sema, &make, None).syntax().clone().into();
            edit.replace_with_many(
                last_element,
                vec![last_stmt.into(), whitespace.into(), early_expr],
            );
        }

        ast::BlockExpr::cast(edit.finish().new_root().clone()).unwrap()
    }
}

enum EarlyKind<'db> {
    Continue,
    Return(hir::Type<'db>),
}

impl<'db> EarlyKind<'db> {
    fn from_node(
        parent_container: &SyntaxNode,
        sema: &Semantics<'db, RootDatabase>,
    ) -> Option<Self> {
        match_ast! {
            match parent_container {
                ast::Fn(it) => Some(Self::Return(sema.to_def(&it)?.ret_type(sema.db))),
                ast::ClosureExpr(it) => Some(Self::Return(sema.type_of_expr(&it.body()?)?.original)),
                ast::WhileExpr(_) => Some(Self::Continue),
                ast::LoopExpr(_) => Some(Self::Continue),
                ast::ForExpr(_) => Some(Self::Continue),
                _ => None
            }
        }
    }

    fn make_early_expr(
        &self,
        sema: &Semantics<'_, RootDatabase>,
        make: &SyntaxFactory,
        ret: Option<ast::Expr>,
    ) -> ast::Expr {
        match self {
            EarlyKind::Continue => make.expr_continue(None).into(),
            EarlyKind::Return(ty) => {
                let expr = match TryEnum::from_ty(sema, ty) {
                    Some(TryEnum::Option) => {
                        ret.or_else(|| Some(make.expr_path(make.ident_path("None"))))
                    }
                    _ => ret,
                };
                make.expr_return(expr).into()
            }
        }
    }

    fn is_unit(&self) -> bool {
        match self {
            EarlyKind::Continue => true,
            EarlyKind::Return(ty) => ty.is_unit(),
        }
    }
}

fn flat_let_chain(mut expr: ast::Expr, make: &SyntaxFactory) -> Vec<ast::Expr> {
    let mut chains = vec![];
    let mut reduce_cond = |rhs| {
        if !matches!(rhs, ast::Expr::LetExpr(_))
            && let Some(last) = chains.pop_if(|last| !matches!(last, ast::Expr::LetExpr(_)))
        {
            chains.push(make.expr_bin_op(rhs, ast::BinaryOp::LogicOp(ast::LogicOp::And), last));
        } else {
            chains.push(rhs);
        }
    };

    while let ast::Expr::BinExpr(bin_expr) = &expr
        && bin_expr.op_kind() == Some(ast::BinaryOp::LogicOp(ast::LogicOp::And))
        && let (Some(lhs), Some(rhs)) = (bin_expr.lhs(), bin_expr.rhs())
    {
        reduce_cond(rhs.reset_indent());
        expr = lhs;
    }

    reduce_cond(expr.reset_indent());
    chains.reverse();
    chains
}

fn clean_stmt_block(block: &ast::BlockExpr, make: &SyntaxFactory) -> ast::BlockExpr {
    if block.statements().next().is_none()
        && let Some(tail_expr) = block.tail_expr()
        && block.modifier().is_none()
    {
        make.block_expr(once(make.expr_stmt(tail_expr).into()), None)
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
    fn convert_if_let_has_else_block() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() -> i32 {
    if$0 true {
        foo();
    } else {
        bar()
    }
}
"#,
            r#"
fn main() -> i32 {
    if false {
        return bar();
    }
    foo();
}
"#,
        );

        check_assist(
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
            r#"
fn main() {
    if false {
        bar();
        return
    }
    foo();
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    if$0 true {
        foo();
    } else {
        bar();
    }
}
"#,
            r#"
fn main() {
    if false {
        bar();
        return
    }
    foo();
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
    fn convert_if_let_has_never_type_else_block_in_statement() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    some_statements();
    if$0 let Ok(x) = Err(92) {
        foo(x);
    } else {
        // needless comment
        return;
    }
    some_statements();
}
"#,
            r#"
fn main() {
    some_statements();
    let Ok(x) = Err(92) else {
        // needless comment
        return;
    };
    foo(x);
    some_statements();
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
    fn convert_let_inside_for_with_else() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    for n in ns {
        if$0 let Some(n) = n {
            foo(n);
            bar();
        } else {
            baz()
        }
    }
}
"#,
            r#"
fn main() {
    for n in ns {
        let Some(n) = n else {
            baz();
            continue
        };
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
    fn convert_let_ref_stmt_inside_fn() {
        check_assist(
            convert_to_guarded_return,
            r#"
//- minicore: option
fn foo() -> &'static Option<i32> {
    &None
}

fn main() {
    let x$0 = foo();
}
"#,
            r#"
fn foo() -> &'static Option<i32> {
    &None
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
    fn indentations() {
        check_assist(
            convert_to_guarded_return,
            r#"
mod indent {
    fn main() {
        $0if let None = Some(
            92
        ) {
            foo(
                93
            );
        }
    }
}
"#,
            r#"
mod indent {
    fn main() {
        let None = Some(
            92
        ) else { return };
        foo(
            93
        );
    }
}
"#,
        );

        check_assist(
            convert_to_guarded_return,
            r#"
//- minicore: option
mod indent {
    fn foo(_: i32) -> Option<i32> { None }
    fn main() {
        $0let x = foo(
            2
        );
    }
}
"#,
            r#"
mod indent {
    fn foo(_: i32) -> Option<i32> { None }
    fn main() {
        let Some(x) = foo(
            2
        ) else { return };
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
    fn ignore_else_branch_has_non_never_types_in_statement() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
fn main() {
    some_statements();
    if$0 true {
        foo();
    } else {
        bar()
    }
    some_statements();
}
"#,
        );
    }

    #[test]
    fn ignore_else_if() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
fn main() {
    some_statements();
    if cond {
        ()
    } else if$0 let Ok(x) = Err(92) {
        foo(x);
    } else {
        return;
    }
    some_statements();
}
"#,
        );
    }

    #[test]
    fn ignore_if_inside_let() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
fn main() {
    some_statements();
    let _ = if$0 let Ok(x) = Err(92) {
        foo(x);
    } else {
        return;
    }
    some_statements();
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
