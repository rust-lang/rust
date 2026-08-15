use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: cannot-be-dereferenced
//
// This diagnostic is triggered if the operand of a dereference expression
// cannot be dereferenced.
pub(crate) fn cannot_be_dereferenced(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::CannotBeDereferenced<'_>,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0614"),
        format!(
            "type `{}` cannot be dereferenced",
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
    fn cannot_be_dereferenced() {
        check_diagnostics(
            r#"
fn f() {
    let x = 1i32;
    let _ = *x;
          //^^ error: type `i32` cannot be dereferenced
}
"#,
        );
    }

    #[test]
    fn allows_reference_deref() {
        check_diagnostics(
            r#"
fn f(x: &i32) {
    let _ = *x;
}
"#,
        );
    }

    #[test]
    fn allows_overloaded_deref() {
        check_diagnostics(
            r#"
//- minicore: deref
struct Wrapper(i32);

impl core::ops::Deref for Wrapper {
    type Target = i32;

    fn deref(&self) -> &i32 {
        &self.0
    }
}

fn f(x: Wrapper) {
    let _ = *x;
}
"#,
        );
    }
}
