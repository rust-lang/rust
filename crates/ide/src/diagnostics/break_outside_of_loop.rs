use crate::diagnostics::{Diagnostic, DiagnosticsContext};

// Diagnostic: break-outside-of-loop
//
// This diagnostic is triggered if the `break` keyword is used outside of a loop.
pub(super) fn break_outside_of_loop(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::BreakOutsideOfLoop,
) -> Diagnostic {
    Diagnostic::new(
        "break-outside-of-loop",
        "break outside of loop",
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics;

    #[test]
    fn break_outside_of_loop() {
        check_diagnostics(
            r#"
fn foo() { break; }
         //^^^^^ break outside of loop
"#,
        );
    }
}
