use hir::db::AstDatabase;
use ide_db::source_change::SourceChange;
use syntax::AstNode;
use text_edit::TextEdit;

use crate::{fix, Assist, Diagnostic, DiagnosticsContext};

// Diagnostic: add-reference-here
//
// This diagnostic is triggered when there's a missing referencing of expression.
pub(crate) fn add_reference_here(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::AddReferenceHere,
) -> Diagnostic {
    Diagnostic::new(
        "add-reference-here",
        "add reference here",
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::AddReferenceHere) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.expr.file_id)?;
    let arg_expr = d.expr.value.to_node(&root);

    let arg_with_ref = format!("&{}{}", d.mutability.as_keyword_for_ref(), arg_expr.syntax());

    let arg_range = arg_expr.syntax().text_range();
    let edit = TextEdit::replace(arg_range, arg_with_ref);
    let source_change =
        SourceChange::from_text_edit(d.expr.file_id.original_file(ctx.sema.db), edit);

    Some(vec![fix("add_reference_here", "Add reference here", source_change, arg_range)])
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
       //^^^ 💡 error: add reference here
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
