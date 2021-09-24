use hir::{db::AstDatabase, TypeInfo};
use ide_db::{assists::Assist, helpers::for_each_tail_expr, source_change::SourceChange};
use syntax::AstNode;
use text_edit::TextEdit;

use crate::{fix, Diagnostic, DiagnosticsContext};

// Diagnostic: missing-ok-or-some-in-tail-expr
//
// This diagnostic is triggered if a block that should return `Result` returns a value not wrapped in `Ok`,
// or if a block that should return `Option` returns a value not wrapped in `Some`.
//
// Example:
//
// ```rust
// fn foo() -> Result<u8, ()> {
//     10
// }
// ```
pub(crate) fn missing_ok_or_some_in_tail_expr(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MissingOkOrSomeInTailExpr,
) -> Diagnostic {
    Diagnostic::new(
        "missing-ok-or-some-in-tail-expr",
        format!("wrap return expression in {}", d.required),
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::MissingOkOrSomeInTailExpr) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.expr.file_id)?;
    let tail_expr = d.expr.value.to_node(&root);
    let tail_expr_range = tail_expr.syntax().text_range();
    let mut builder = TextEdit::builder();
    for_each_tail_expr(&tail_expr, &mut |expr| {
        if ctx.sema.type_of_expr(expr).map(TypeInfo::original).as_ref() != Some(&d.expected) {
            builder.insert(expr.syntax().text_range().start(), format!("{}(", d.required));
            builder.insert(expr.syntax().text_range().end(), ")".to_string());
        }
    });
    let source_change =
        SourceChange::from_text_edit(d.expr.file_id.original_file(ctx.sema.db), builder.finish());
    let name = if d.required == "Ok" { "Wrap with Ok" } else { "Wrap with Some" };
    Some(vec![fix("wrap_tail_expr", name, source_change, tail_expr_range)])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn test_wrap_return_type_option() {
        check_fix(
            r#"
//- minicore: option, result
fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        return None;
    }
    x / y$0
}
"#,
            r#"
fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        return None;
    }
    Some(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_option_tails() {
        check_fix(
            r#"
//- minicore: option, result
fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        0
    } else if true {
        100
    } else {
        None
    }$0
}
"#,
            r#"
fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        Some(0)
    } else if true {
        Some(100)
    } else {
        None
    }
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type() {
        check_fix(
            r#"
//- minicore: option, result
fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    x / y$0
}
"#,
            r#"
fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    Ok(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_handles_generic_functions() {
        check_fix(
            r#"
//- minicore: option, result
fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    $0x
}
"#,
            r#"
fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    Ok(x)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_handles_type_aliases() {
        check_fix(
            r#"
//- minicore: option, result
type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    x $0/ y
}
"#,
            r#"
type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    Ok(x / y)
}
"#,
        );
    }

    #[test]
    fn test_in_const_and_static() {
        check_fix(
            r#"
//- minicore: option, result
static A: Option<()> = {($0)};
            "#,
            r#"
static A: Option<()> = {Some(())};
            "#,
        );
        check_fix(
            r#"
//- minicore: option, result
const _: Option<()> = {($0)};
            "#,
            r#"
const _: Option<()> = {Some(())};
            "#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_expr_type_does_not_match_ok_type() {
        check_diagnostics(
            r#"
//- minicore: option, result
fn foo() -> Result<(), i32> { 0 }
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_return_type_is_not_result_or_option() {
        check_diagnostics(
            r#"
//- minicore: option, result
enum SomeOtherEnum { Ok(i32), Err(String) }

fn foo() -> SomeOtherEnum { 0 }
"#,
        );
    }
}
