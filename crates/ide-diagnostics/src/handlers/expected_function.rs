use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: expected-function
//
// This diagnostic is triggered if a call is made on something that is not callable.
pub(crate) fn expected_function(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::ExpectedFunction,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0618"),
        format!("expected function, found {}", d.found.display(ctx.sema.db)),
        d.call.clone().map(|it| it.into()),
    )
    .experimental()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn smoke_test() {
        check_diagnostics(
            r#"
fn foo() {
    let x = 3;
    x();
 // ^^^ error: expected function, found i32
    ""();
 // ^^^^ error: expected function, found &str
    foo();
}
"#,
        );
    }
}
