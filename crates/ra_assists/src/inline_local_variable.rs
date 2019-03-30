use hir::db::HirDatabase;
use hir::source_binder::function_from_child_node;
use ra_syntax::{ast::{self, AstNode}, TextRange};
use ra_syntax::ast::{PatKind, ExprKind};

use crate::{Assist, AssistCtx, AssistId};
use crate::assist_ctx::AssistBuilder;

pub(crate) fn inline_local_varialbe(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let let_stmt = ctx.node_at_offset::<ast::LetStmt>()?;
    let bind_pat = match let_stmt.pat()?.kind() {
        PatKind::BindPat(pat) => pat,
        _ => return None,
    };
    if bind_pat.is_mutable() {
        return None;
    }
    let initializer = let_stmt.initializer()?;
    let wrap_in_parens = match initializer.kind() {
        ExprKind::LambdaExpr(_)
        | ExprKind::IfExpr(_)
        | ExprKind::LoopExpr(_)
        | ExprKind::ForExpr(_)
        | ExprKind::WhileExpr(_)
        | ExprKind::ContinueExpr(_)
        | ExprKind::BreakExpr(_)
        | ExprKind::Label(_)
        | ExprKind::ReturnExpr(_)
        | ExprKind::MatchExpr(_)
        | ExprKind::StructLit(_)
        | ExprKind::CastExpr(_)
        | ExprKind::PrefixExpr(_)
        | ExprKind::RangeExpr(_)
        | ExprKind::BinExpr(_) => true,
        ExprKind::CallExpr(_)
        | ExprKind::IndexExpr(_)
        | ExprKind::MethodCallExpr(_)
        | ExprKind::FieldExpr(_)
        | ExprKind::TryExpr(_)
        | ExprKind::RefExpr(_)
        | ExprKind::Literal(_)
        | ExprKind::TupleExpr(_)
        | ExprKind::ArrayExpr(_)
        | ExprKind::ParenExpr(_)
        | ExprKind::PathExpr(_)
        | ExprKind::BlockExpr(_) => false,
    };

    let delete_range = if let Some(whitespace) = let_stmt
        .syntax()
        .next_sibling_or_token()
        .and_then(|it| ast::Whitespace::cast(it.as_token()?))
    {
        TextRange::from_to(let_stmt.syntax().range().start(), whitespace.syntax().range().end())
    } else {
        let_stmt.syntax().range()
    };

    let init_str = if wrap_in_parens {
        format!("({})", initializer.syntax().text().to_string())
    } else {
        initializer.syntax().text().to_string()
    };
    let function = function_from_child_node(ctx.db, ctx.frange.file_id, bind_pat.syntax())?;
    let scope = function.scopes(ctx.db);
    let refs = scope.find_all_refs(bind_pat);

    ctx.add_action(
        AssistId("inline_local_variable"),
        "inline local variable",
        move |edit: &mut AssistBuilder| {
            edit.delete(delete_range);
            for desc in refs {
                edit.replace(desc.range, init_str.clone())
            }
            edit.set_cursor(delete_range.start())
        },
    );

    ctx.build()
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_inline_let_bind_literal_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn bar(a: usize) {}
fn foo() {
    let a<|> = 1;
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            "
fn bar(a: usize) {}
fn foo() {
    <|>1 + 1;
    if 1 > 10 {
    }

    while 1 > 10 {

    }
    let b = 1 * 10;
    bar(1);
}",
        );
    }

    #[test]
    fn test_inline_let_bind_bin_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn bar(a: usize) {}
fn foo() {
    let a<|> = 1 + 1;
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            "
fn bar(a: usize) {}
fn foo() {
    <|>(1 + 1) + 1;
    if (1 + 1) > 10 {
    }

    while (1 + 1) > 10 {

    }
    let b = (1 + 1) * 10;
    bar((1 + 1));
}",
        );
    }

    #[test]
    fn test_inline_let_bind_function_call_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn bar(a: usize) {}
fn foo() {
    let a<|> = bar(1);
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            "
fn bar(a: usize) {}
fn foo() {
    <|>bar(1) + 1;
    if bar(1) > 10 {
    }

    while bar(1) > 10 {

    }
    let b = bar(1) * 10;
    bar(bar(1));
}",
        );
    }

    #[test]
    fn test_inline_let_bind_cast_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn bar(a: usize): usize { a }
fn foo() {
    let a<|> = bar(1) as u64;
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            "
fn bar(a: usize): usize { a }
fn foo() {
    <|>(bar(1) as u64) + 1;
    if (bar(1) as u64) > 10 {
    }

    while (bar(1) as u64) > 10 {

    }
    let b = (bar(1) as u64) * 10;
    bar((bar(1) as u64));
}",
        );
    }

    #[test]
    fn test_inline_let_bind_block_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = { 10 + 1 };
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            "
fn foo() {
    <|>{ 10 + 1 } + 1;
    if { 10 + 1 } > 10 {
    }

    while { 10 + 1 } > 10 {

    }
    let b = { 10 + 1 } * 10;
    bar({ 10 + 1 });
}",
        );
    }

    #[test]
    fn test_inline_let_bind_paren_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = ( 10 + 1 );
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            "
fn foo() {
    <|>( 10 + 1 ) + 1;
    if ( 10 + 1 ) > 10 {
    }

    while ( 10 + 1 ) > 10 {

    }
    let b = ( 10 + 1 ) * 10;
    bar(( 10 + 1 ));
}",
        );
    }

    #[test]
    fn test_not_inline_mut_variable() {
        check_assist_not_applicable(
            inline_local_varialbe,
            "
fn foo() {
    let mut a<|> = 1 + 1;
    a + 1;
}",
        );
    }
}
