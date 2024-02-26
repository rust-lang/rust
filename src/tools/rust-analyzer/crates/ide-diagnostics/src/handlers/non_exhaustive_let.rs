use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: non-exhaustive-let
//
// This diagnostic is triggered if a `let` statement without an `else` branch has a non-exhaustive
// pattern.
pub(crate) fn non_exhaustive_let(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::NonExhaustiveLet,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0005"),
        format!("non-exhaustive pattern: {}", d.uncovered_patterns),
        d.pat.map(Into::into),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn option_nonexhaustive() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    let None = Some(5);
      //^^^^ error: non-exhaustive pattern: `Some(_)` not covered
}
"#,
        );
    }

    #[test]
    fn option_exhaustive() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    let Some(_) | None = Some(5);
}
"#,
        );
    }
}
