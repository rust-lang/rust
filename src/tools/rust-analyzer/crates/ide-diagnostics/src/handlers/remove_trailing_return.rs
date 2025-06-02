use hir::{FileRange, db::ExpandDatabase, diagnostics::RemoveTrailingReturn};
use ide_db::text_edit::TextEdit;
use ide_db::{assists::Assist, source_change::SourceChange};
use syntax::{AstNode, ast};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, adjusted_display_range, fix};

// Diagnostic: remove-trailing-return
//
// This diagnostic is triggered when there is a redundant `return` at the end of a function
// or closure.
pub(crate) fn remove_trailing_return(
    ctx: &DiagnosticsContext<'_>,
    d: &RemoveTrailingReturn,
) -> Option<Diagnostic> {
    if d.return_expr.file_id.macro_file().is_some() {
        // FIXME: Our infra can't handle allow from within macro expansions rn
        return None;
    }

    let display_range = adjusted_display_range(ctx, d.return_expr, &|return_expr| {
        return_expr
            .syntax()
            .parent()
            .and_then(ast::ExprStmt::cast)
            .map(|stmt| stmt.syntax().text_range())
    });
    Some(
        Diagnostic::new(
            DiagnosticCode::Clippy("needless_return"),
            "replace return <expr>; with <expr>",
            display_range,
        )
        .stable()
        .with_fixes(fixes(ctx, d)),
    )
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &RemoveTrailingReturn) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.return_expr.file_id);
    let return_expr = d.return_expr.value.to_node(&root);
    let stmt = return_expr.syntax().parent().and_then(ast::ExprStmt::cast);

    let FileRange { range, file_id } =
        ctx.sema.original_range_opt(stmt.as_ref().map_or(return_expr.syntax(), AstNode::syntax))?;
    if Some(file_id) != d.return_expr.file_id.file_id() {
        return None;
    }

    let replacement =
        return_expr.expr().map_or_else(String::new, |expr| format!("{}", expr.syntax().text()));
    let edit = TextEdit::replace(range, replacement);
    let source_change = SourceChange::from_text_edit(file_id.file_id(ctx.sema.db), edit);

    Some(vec![fix(
        "remove_trailing_return",
        "Replace return <expr>; with <expr>",
        source_change,
        range,
    )])
}

#[cfg(test)]
mod tests {
    use crate::tests::{
        check_diagnostics, check_diagnostics_with_disabled, check_fix, check_fix_with_disabled,
    };

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
        check_diagnostics_with_disabled(
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
            &["remove-unnecessary-else"],
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
    fn no_diagnostic_if_not_last_statement2() {
        check_diagnostics(
            r#"
fn foo() -> u8 {
    return 2;
    fn bar() {}
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
        check_fix_with_disabled(
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
            std::iter::once("remove-unnecessary-else".to_owned()),
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
