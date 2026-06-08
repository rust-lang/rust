use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: array-pattern-without-fixed-length
//
// This diagnostic is triggered when a rest array pattern is matched against an
// array with a non-constant length.
pub(crate) fn array_pattern_without_fixed_length(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::ArrayPatternWithoutFixedLength,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0730"),
        "cannot pattern-match on an array without a fixed length",
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn array_pattern_without_fixed_length() {
        check_diagnostics(
            r#"
fn f<const N: usize>(arr: [u8; N]) {
    let [_head, _tail @ ..] = arr;
      //^^^^^^^^^^^^^^^^^^^ error: cannot pattern-match on an array without a fixed length
}
"#,
        );
    }

    #[test]
    fn fixed_length_array_pattern_is_ok() {
        check_diagnostics(
            r#"
fn f(arr: [u8; 3]) {
    let [_head, _tail @ ..] = arr;
}
"#,
        );
    }
}
