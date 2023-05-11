use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: expected-function
//
// This diagnostic is triggered if a call is made on something that is not callable.
pub(crate) fn expected_function(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::ExpectedFunction,
) -> Diagnostic {
    Diagnostic::new(
        "expected-function",
        format!("expected function, found {}", d.found.display(ctx.sema.db)),
        ctx.sema.diagnostics_display_range(d.call.clone().map(|it| it.into())).range,
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
