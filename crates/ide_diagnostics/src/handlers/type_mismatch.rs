use hir::{db::AstDatabase, HirDisplay, Type};
use ide_db::source_change::SourceChange;
use syntax::{AstNode, TextRange};
use text_edit::TextEdit;

use crate::{fix, Assist, Diagnostic, DiagnosticsContext};

// Diagnostic: type-mismatch
//
// This diagnostic is triggered when the type of an expression does not match
// the expected type.
pub(crate) fn type_mismatch(ctx: &DiagnosticsContext<'_>, d: &hir::TypeMismatch) -> Diagnostic {
    let mut diag = Diagnostic::new(
        "type-mismatch",
        format!(
            "expected {}, found {}",
            d.expected.display(ctx.sema.db),
            d.actual.display(ctx.sema.db)
        ),
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d));
    if diag.fixes.is_none() {
        diag.experimental = true;
    }
    diag
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::TypeMismatch) -> Option<Vec<Assist>> {
    let mut fixes = Vec::new();

    add_reference(ctx, d, &mut fixes);

    if fixes.is_empty() {
        None
    } else {
        Some(fixes)
    }
}

fn add_reference(ctx: &DiagnosticsContext<'_>, d: &hir::TypeMismatch, acc: &mut Vec<Assist>) -> Option<()> {
    let root = ctx.sema.db.parse_or_expand(d.expr.file_id)?;
    let expr_node = d.expr.value.to_node(&root);

    let range = ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range;

    let (_, mutability) = d.expected.as_reference()?;
    let actual_with_ref = Type::reference(&d.actual, mutability);
    if !actual_with_ref.could_coerce_to(ctx.sema.db, &d.expected) {
        return None;
    }

    let ampersands = format!("&{}", mutability.as_keyword_for_ref());

    let edit = TextEdit::insert(expr_node.syntax().text_range().start(), ampersands);
    let source_change =
        SourceChange::from_text_edit(d.expr.file_id.original_file(ctx.sema.db), edit);
    acc.push(fix("add_reference_here", "Add reference here", source_change, range));
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn missing_reference() {
        check_diagnostics(
            r#"
fn main() {
    test(123);
       //^^^ 💡 error: expected &i32, found i32
}
fn test(arg: &i32) {}
"#,
        );
    }

    #[test]
    fn test_add_reference_to_int() {
        check_fix(
            r#"
fn main() {
    test(123$0);
}
fn test(arg: &i32) {}
            "#,
            r#"
fn main() {
    test(&123);
}
fn test(arg: &i32) {}
            "#,
        );
    }

    #[test]
    fn test_add_mutable_reference_to_int() {
        check_fix(
            r#"
fn main() {
    test($0123);
}
fn test(arg: &mut i32) {}
            "#,
            r#"
fn main() {
    test(&mut 123);
}
fn test(arg: &mut i32) {}
            "#,
        );
    }

    #[test]
    fn test_add_reference_to_array() {
        check_fix(
            r#"
//- minicore: coerce_unsized
fn main() {
    test($0[1, 2, 3]);
}
fn test(arg: &[i32]) {}
            "#,
            r#"
fn main() {
    test(&[1, 2, 3]);
}
fn test(arg: &[i32]) {}
            "#,
        );
    }

    #[test]
    fn test_add_reference_with_autoderef() {
        check_fix(
            r#"
//- minicore: coerce_unsized, deref
struct Foo;
struct Bar;
impl core::ops::Deref for Foo {
    type Target = Bar;
}

fn main() {
    test($0Foo);
}
fn test(arg: &Bar) {}
            "#,
            r#"
struct Foo;
struct Bar;
impl core::ops::Deref for Foo {
    type Target = Bar;
}

fn main() {
    test(&Foo);
}
fn test(arg: &Bar) {}
            "#,
        );
    }

    #[test]
    fn test_add_reference_to_method_call() {
        check_fix(
            r#"
fn main() {
    Test.call_by_ref($0123);
}
struct Test;
impl Test {
    fn call_by_ref(&self, arg: &i32) {}
}
            "#,
            r#"
fn main() {
    Test.call_by_ref(&123);
}
struct Test;
impl Test {
    fn call_by_ref(&self, arg: &i32) {}
}
            "#,
        );
    }

    #[test]
    fn test_add_reference_to_let_stmt() {
        check_fix(
            r#"
fn main() {
    let test: &i32 = $0123;
}
            "#,
            r#"
fn main() {
    let test: &i32 = &123;
}
            "#,
        );
    }

    #[test]
    fn test_add_mutable_reference_to_let_stmt() {
        check_fix(
            r#"
fn main() {
    let test: &mut i32 = $0123;
}
            "#,
            r#"
fn main() {
    let test: &mut i32 = &mut 123;
}
            "#,
        );
    }
}
