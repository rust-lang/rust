use hir::db::HirDatabase;
use ra_syntax::{
    ast::{self, AstNode, AstToken},
    TextRange,
};

use crate::assist_ctx::AssistBuilder;
use crate::{Assist, AssistCtx, AssistId};

// Assist: inline_local_variable
//
// Inlines local variable.
//
// ```
// fn main() {
//     let x<|> = 1 + 2;
//     x * 4;
// }
// ```
// ->
// ```
// fn main() {
//     (1 + 2) * 4;
// }
// ```
pub(crate) fn inline_local_varialbe(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let let_stmt = ctx.find_node_at_offset::<ast::LetStmt>()?;
    let bind_pat = match let_stmt.pat()? {
        ast::Pat::BindPat(pat) => pat,
        _ => return None,
    };
    if bind_pat.is_mutable() {
        return None;
    }
    let initializer_expr = let_stmt.initializer()?;
    let delete_range = if let Some(whitespace) = let_stmt
        .syntax()
        .next_sibling_or_token()
        .and_then(|it| ast::Whitespace::cast(it.as_token()?.clone()))
    {
        TextRange::from_to(
            let_stmt.syntax().text_range().start(),
            whitespace.syntax().text_range().end(),
        )
    } else {
        let_stmt.syntax().text_range()
    };
    let analyzer = ctx.source_analyzer(bind_pat.syntax(), None);
    let refs = analyzer.find_all_refs(&bind_pat);

    let mut wrap_in_parens = vec![true; refs.len()];

    for (i, desc) in refs.iter().enumerate() {
        let usage_node =
            ctx.covering_node_for_range(desc.range).ancestors().find_map(ast::PathExpr::cast)?;
        let usage_parent_option = usage_node.syntax().parent().and_then(ast::Expr::cast);
        let usage_parent = match usage_parent_option {
            Some(u) => u,
            None => {
                wrap_in_parens[i] = false;
                continue;
            }
        };

        wrap_in_parens[i] = match (&initializer_expr, usage_parent) {
            (ast::Expr::CallExpr(_), _)
            | (ast::Expr::IndexExpr(_), _)
            | (ast::Expr::MethodCallExpr(_), _)
            | (ast::Expr::FieldExpr(_), _)
            | (ast::Expr::TryExpr(_), _)
            | (ast::Expr::RefExpr(_), _)
            | (ast::Expr::Literal(_), _)
            | (ast::Expr::TupleExpr(_), _)
            | (ast::Expr::ArrayExpr(_), _)
            | (ast::Expr::ParenExpr(_), _)
            | (ast::Expr::PathExpr(_), _)
            | (ast::Expr::BlockExpr(_), _)
            | (_, ast::Expr::CallExpr(_))
            | (_, ast::Expr::TupleExpr(_))
            | (_, ast::Expr::ArrayExpr(_))
            | (_, ast::Expr::ParenExpr(_))
            | (_, ast::Expr::ForExpr(_))
            | (_, ast::Expr::WhileExpr(_))
            | (_, ast::Expr::BreakExpr(_))
            | (_, ast::Expr::ReturnExpr(_))
            | (_, ast::Expr::MatchExpr(_)) => false,
            _ => true,
        };
    }

    let init_str = initializer_expr.syntax().text().to_string();
    let init_in_paren = format!("({})", &init_str);

    ctx.add_assist(
        AssistId("inline_local_variable"),
        "inline local variable",
        move |edit: &mut AssistBuilder| {
            edit.delete(delete_range);
            for (desc, should_wrap) in refs.iter().zip(wrap_in_parens) {
                if should_wrap {
                    edit.replace(desc.range, init_in_paren.clone())
                } else {
                    edit.replace(desc.range, init_str.clone())
                }
            }
            edit.set_cursor(delete_range.start())
        },
    )
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
    bar(1 + 1);
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
    bar(bar(1) as u64);
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

    #[test]
    fn test_call_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = bar(10 + 1);
    let b = a * 10;
    let c = a as usize;
}",
            "
fn foo() {
    <|>let b = bar(10 + 1) * 10;
    let c = bar(10 + 1) as usize;
}",
        );
    }

    #[test]
    fn test_index_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let x = vec![1, 2, 3];
    let a<|> = x[0];
    let b = a * 10;
    let c = a as usize;
}",
            "
fn foo() {
    let x = vec![1, 2, 3];
    <|>let b = x[0] * 10;
    let c = x[0] as usize;
}",
        );
    }

    #[test]
    fn test_method_call_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let bar = vec![1];
    let a<|> = bar.len();
    let b = a * 10;
    let c = a as usize;
}",
            "
fn foo() {
    let bar = vec![1];
    <|>let b = bar.len() * 10;
    let c = bar.len() as usize;
}",
        );
    }

    #[test]
    fn test_field_expr() {
        check_assist(
            inline_local_varialbe,
            "
struct Bar {
    foo: usize
}

fn foo() {
    let bar = Bar { foo: 1 };
    let a<|> = bar.foo;
    let b = a * 10;
    let c = a as usize;
}",
            "
struct Bar {
    foo: usize
}

fn foo() {
    let bar = Bar { foo: 1 };
    <|>let b = bar.foo * 10;
    let c = bar.foo as usize;
}",
        );
    }

    #[test]
    fn test_try_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() -> Option<usize> {
    let bar = Some(1);
    let a<|> = bar?;
    let b = a * 10;
    let c = a as usize;
    None
}",
            "
fn foo() -> Option<usize> {
    let bar = Some(1);
    <|>let b = bar? * 10;
    let c = bar? as usize;
    None
}",
        );
    }

    #[test]
    fn test_ref_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let bar = 10;
    let a<|> = &bar;
    let b = a * 10;
}",
            "
fn foo() {
    let bar = 10;
    <|>let b = &bar * 10;
}",
        );
    }

    #[test]
    fn test_tuple_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = (10, 20);
    let b = a[0];
}",
            "
fn foo() {
    <|>let b = (10, 20)[0];
}",
        );
    }

    #[test]
    fn test_array_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = [1, 2, 3];
    let b = a.len();
}",
            "
fn foo() {
    <|>let b = [1, 2, 3].len();
}",
        );
    }

    #[test]
    fn test_paren() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = (10 + 20);
    let b = a * 10;
    let c = a as usize;
}",
            "
fn foo() {
    <|>let b = (10 + 20) * 10;
    let c = (10 + 20) as usize;
}",
        );
    }

    #[test]
    fn test_path_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let d = 10;
    let a<|> = d;
    let b = a * 10;
    let c = a as usize;
}",
            "
fn foo() {
    let d = 10;
    <|>let b = d * 10;
    let c = d as usize;
}",
        );
    }

    #[test]
    fn test_block_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = { 10 };
    let b = a * 10;
    let c = a as usize;
}",
            "
fn foo() {
    <|>let b = { 10 } * 10;
    let c = { 10 } as usize;
}",
        );
    }

    #[test]
    fn test_used_in_different_expr1() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = 10 + 20;
    let b = a * 10;
    let c = (a, 20);
    let d = [a, 10];
    let e = (a);
}",
            "
fn foo() {
    <|>let b = (10 + 20) * 10;
    let c = (10 + 20, 20);
    let d = [10 + 20, 10];
    let e = (10 + 20);
}",
        );
    }

    #[test]
    fn test_used_in_for_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = vec![10, 20];
    for i in a {}
}",
            "
fn foo() {
    <|>for i in vec![10, 20] {}
}",
        );
    }

    #[test]
    fn test_used_in_while_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = 1 > 0;
    while a {}
}",
            "
fn foo() {
    <|>while 1 > 0 {}
}",
        );
    }

    #[test]
    fn test_used_in_break_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = 1 + 1;
    loop {
        break a;
    }
}",
            "
fn foo() {
    <|>loop {
        break 1 + 1;
    }
}",
        );
    }

    #[test]
    fn test_used_in_return_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = 1 > 0;
    return a;
}",
            "
fn foo() {
    <|>return 1 > 0;
}",
        );
    }

    #[test]
    fn test_used_in_match_expr() {
        check_assist(
            inline_local_varialbe,
            "
fn foo() {
    let a<|> = 1 > 0;
    match a {}
}",
            "
fn foo() {
    <|>match 1 > 0 {}
}",
        );
    }
}
