use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: yield-outside-of-coroutine
//
// This diagnostic is triggered if the `yield` keyword is used outside of a coroutine.
pub(crate) fn yield_outside_coroutine(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::YieldOutsideCoroutine,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0627"),
        "yield expression outside of coroutine",
        d.expr.map(|it| it.into()),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn yield_in_regular_function() {
        check_diagnostics(
            r#"
//- minicore: coroutine
fn foo() {
    yield 1;
  //^^^^^^^ error: yield expression outside of coroutine
}
"#,
        );
    }
}
