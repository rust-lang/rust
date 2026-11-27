use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: invalid-range-pat-type
//
// This diagnostic is triggered when a range pattern is used with a type that
// is neither `char` nor numeric.
pub(crate) fn invalid_range_pat_type(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::InvalidRangePatType,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0029"),
        "only `char` and numeric types are allowed in range patterns",
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn bool_range_pattern() {
        check_diagnostics(
            r#"
fn f(x: bool) {
    match x {
        false..=true => {}
      //^^^^^^^^^^^^ error: only `char` and numeric types are allowed in range patterns
    }
}
"#,
        );
    }

    #[test]
    fn numeric_and_char_range_patterns() {
        check_diagnostics(
            r#"
fn f(x: u8, c: char) {
    match x {
        0..=9 => {}
        _ => {}
    }
    match c {
        'a'..='z' => {}
        _ => {}
    }
}
"#,
        );
    }
}
