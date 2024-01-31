use hir::{db::ExpandDatabase, diagnostics::RemoveTrailingReturn, HirFileIdExt, InFile};
use ide_db::{assists::Assist, source_change::SourceChange};
use syntax::{ast, AstNode, SyntaxNodePtr};
use text_edit::TextEdit;

use crate::{fix, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: remove-trailing-return
//
// This diagnostic is triggered when there is a redundant `return` at the end of a function
// or closure.
pub(crate) fn remove_trailing_return(
    ctx: &DiagnosticsContext<'_>,
    d: &RemoveTrailingReturn,
) -> Diagnostic {
    let display_range = ctx.sema.diagnostics_display_range(InFile {
        file_id: d.file_id,
        value: expr_stmt(ctx, d)
            .as_ref()
            .map(|stmt| SyntaxNodePtr::new(stmt.syntax()))
            .unwrap_or_else(|| d.return_expr.into()),
    });
    Diagnostic::new(
        DiagnosticCode::Clippy("needless_return"),
        "replace return <expr>; with <expr>",
        display_range,
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &RemoveTrailingReturn) -> Option<Vec<Assist>> {
    let return_expr = return_expr(ctx, d)?;
    let stmt = expr_stmt(ctx, d);

    let range = stmt.as_ref().map_or(return_expr.syntax(), AstNode::syntax).text_range();
    let replacement =
        return_expr.expr().map_or_else(String::new, |expr| format!("{}", expr.syntax().text()));

    let edit = TextEdit::replace(range, replacement);
    let source_change = SourceChange::from_text_edit(d.file_id.original_file(ctx.sema.db), edit);

    Some(vec![fix(
        "remove_trailing_return",
        "Replace return <expr>; with <expr>",
        source_change,
        range,
    )])
}

fn return_expr(ctx: &DiagnosticsContext<'_>, d: &RemoveTrailingReturn) -> Option<ast::ReturnExpr> {
    let root = ctx.sema.db.parse_or_expand(d.file_id);
    let expr = d.return_expr.to_node(&root);
    ast::ReturnExpr::cast(expr.syntax().clone())
}

fn expr_stmt(ctx: &DiagnosticsContext<'_>, d: &RemoveTrailingReturn) -> Option<ast::ExprStmt> {
    let return_expr = return_expr(ctx, d)?;
    return_expr.syntax().parent().and_then(ast::ExprStmt::cast)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn remove_trailing_return() {
        check_diagnostics(
            r#"
fn foo() -> u8 {
    return 2;
} //^^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
"#,
        );
    }

    #[test]
    fn remove_trailing_return_inner_function() {
        check_diagnostics(
            r#"
fn foo() -> u8 {
    fn bar() -> u8 {
        return 2;
    } //^^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
    bar()
}
"#,
        );
    }

    #[test]
    fn remove_trailing_return_closure() {
        check_diagnostics(
            r#"
fn foo() -> u8 {
    let bar = || return 2;
    bar()      //^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
}
"#,
        );
        check_diagnostics(
            r#"
fn foo() -> u8 {
    let bar = || {
        return 2;
    };//^^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
    bar()
}
"#,
        );
    }

    #[test]
    fn remove_trailing_return_unit() {
        check_diagnostics(
            r#"
fn foo() {
    return
} //^^^^^^ 💡 weak: replace return <expr>; with <expr>
"#,
        );
    }

    #[test]
    fn remove_trailing_return_no_semi() {
        check_diagnostics(
            r#"
fn foo() -> u8 {
    return 2
} //^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
"#,
        );
    }

    #[test]
    fn remove_trailing_return_in_if() {
        check_diagnostics(
            r#"
fn foo(x: usize) -> u8 {
    if x > 0 {
        return 1;
      //^^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
    } else {
        return 0;
    } //^^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
}
"#,
        );
    }

    #[test]
    fn remove_trailing_return_in_match() {
        check_diagnostics(
            r#"
fn foo<T, E>(x: Result<T, E>) -> u8 {
    match x {
        Ok(_) => return 1,
               //^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
        Err(_) => return 0,
    }           //^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
}
"#,
        );
    }

    #[test]
    fn no_diagnostic_if_no_return_keyword() {
        check_diagnostics(
            r#"
fn foo() -> u8 {
    3
}
"#,
        );
    }

    #[test]
    fn no_diagnostic_if_not_last_statement() {
        check_diagnostics(
            r#"
fn foo() -> u8 {
    if true { return 2; }
    3
}
"#,
        );
    }

    #[test]
    fn replace_with_expr() {
        check_fix(
            r#"
fn foo() -> u8 {
    return$0 2;
}
"#,
            r#"
fn foo() -> u8 {
    2
}
"#,
        );
    }

    #[test]
    fn replace_with_unit() {
        check_fix(
            r#"
fn foo() {
    return$0/*ensure tidy is happy*/
}
"#,
            r#"
fn foo() {
    /*ensure tidy is happy*/
}
"#,
        );
    }

    #[test]
    fn replace_with_expr_no_semi() {
        check_fix(
            r#"
fn foo() -> u8 {
    return$0 2
}
"#,
            r#"
fn foo() -> u8 {
    2
}
"#,
        );
    }

    #[test]
    fn replace_in_inner_function() {
        check_fix(
            r#"
fn foo() -> u8 {
    fn bar() -> u8 {
        return$0 2;
    }
    bar()
}
"#,
            r#"
fn foo() -> u8 {
    fn bar() -> u8 {
        2
    }
    bar()
}
"#,
        );
    }

    #[test]
    fn replace_in_closure() {
        check_fix(
            r#"
fn foo() -> u8 {
    let bar = || return$0 2;
    bar()
}
"#,
            r#"
fn foo() -> u8 {
    let bar = || 2;
    bar()
}
"#,
        );
        check_fix(
            r#"
fn foo() -> u8 {
    let bar = || {
        return$0 2;
    };
    bar()
}
"#,
            r#"
fn foo() -> u8 {
    let bar = || {
        2
    };
    bar()
}
"#,
        );
    }

    #[test]
    fn replace_in_if() {
        check_fix(
            r#"
fn foo(x: usize) -> u8 {
    if x > 0 {
        return$0 1;
    } else {
        0
    }
}
"#,
            r#"
fn foo(x: usize) -> u8 {
    if x > 0 {
        1
    } else {
        0
    }
}
"#,
        );
        check_fix(
            r#"
fn foo(x: usize) -> u8 {
    if x > 0 {
        1
    } else {
        return$0 0;
    }
}
"#,
            r#"
fn foo(x: usize) -> u8 {
    if x > 0 {
        1
    } else {
        0
    }
}
"#,
        );
    }

    #[test]
    fn replace_in_match() {
        check_fix(
            r#"
fn foo<T, E>(x: Result<T, E>) -> u8 {
    match x {
        Ok(_) => return$0 1,
        Err(_) => 0,
    }
}
"#,
            r#"
fn foo<T, E>(x: Result<T, E>) -> u8 {
    match x {
        Ok(_) => 1,
        Err(_) => 0,
    }
}
"#,
        );
        check_fix(
            r#"
fn foo<T, E>(x: Result<T, E>) -> u8 {
    match x {
        Ok(_) => 1,
        Err(_) => return$0 0,
    }
}
"#,
            r#"
fn foo<T, E>(x: Result<T, E>) -> u8 {
    match x {
        Ok(_) => 1,
        Err(_) => 0,
    }
}
"#,
        );
    }
}
