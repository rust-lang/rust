use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: cannot-index-into
//
// This diagnostic is triggered if indexing is used on a type that cannot be
// indexed.
pub(crate) fn cannot_index_into(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::CannotIndexInto<'_>,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0608"),
        format!(
            "cannot index into a value of type `{}`",
            d.found.display(ctx.sema.db, ctx.display_target)
        ),
        d.expr.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn cannot_index_into() {
        check_diagnostics(
            r#"
//- minicore: index
fn f() {
    let x = 1i32;
    let _ = x[0];
          //^^^^ error: cannot index into a value of type `i32`
}
"#,
        );
    }

    #[test]
    fn allows_array_indexing() {
        check_diagnostics(
            r#"
//- minicore: index, slice
fn f() {
    let x = [1i32, 2];
    let _ = x[0];
}
"#,
        );
    }

    #[test]
    fn allows_overloaded_indexing() {
        check_diagnostics(
            r#"
//- minicore: index
struct Bag(i32);

impl core::ops::Index<usize> for Bag {
    type Output = i32;

    fn index(&self, _: usize) -> &Self::Output {
        &self.0
    }
}

fn f(x: Bag) {
    let _ = x[0];
}
"#,
        );
    }
}
