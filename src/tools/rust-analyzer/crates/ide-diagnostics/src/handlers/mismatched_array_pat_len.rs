use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: mismatched-array-pat-len
//
// This diagnostic is triggered when an array pattern's element count does not
// match the array's declared length.
pub(crate) fn mismatched_array_pat_len(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::MismatchedArrayPatLen,
) -> Diagnostic {
    let (code, message) = if d.has_rest {
        (
            "E0528",
            format!(
                "pattern requires at least {} element{} but array has {}",
                d.found,
                if d.found == 1 { "" } else { "s" },
                d.expected,
            ),
        )
    } else {
        (
            "E0527",
            format!(
                "pattern requires {} element{} but array has {}",
                d.found,
                if d.found == 1 { "" } else { "s" },
                d.expected,
            ),
        )
    };
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError(code),
        message,
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn array_pattern_too_few_elements() {
        check_diagnostics(
            r#"
fn f(arr: [i32; 3]) {
    let [_a, _b] = arr;
      //^^^^^^^^ error: pattern requires 2 elements but array has 3
}
"#,
        );
    }

    #[test]
    fn array_pattern_too_many_elements() {
        check_diagnostics(
            r#"
fn f(arr: [i32; 2]) {
    let [_a, _b, _c] = arr;
      //^^^^^^^^^^^^ error: pattern requires 3 elements but array has 2
}
"#,
        );
    }

    #[test]
    fn array_pattern_with_rest_too_short() {
        check_diagnostics(
            r#"
fn f(arr: [i32; 2]) {
    let [_a, _b, _c, ..] = arr;
      //^^^^^^^^^^^^^^^^ error: pattern requires at least 3 elements but array has 2
}
"#,
        );
    }

    #[test]
    fn array_pattern_with_rest_ok() {
        check_diagnostics(
            r#"
fn f(arr: [i32; 5]) {
    let [_a, _b, ..] = arr;
}
"#,
        );
    }

    #[test]
    fn array_pattern_exact_length_ok() {
        check_diagnostics(
            r#"
fn f(arr: [i32; 3]) {
    let [_a, _b, _c] = arr;
}
"#,
        );
    }

    #[test]
    fn array_pattern_singular_element_uses_singular() {
        check_diagnostics(
            r#"
fn f(arr: [i32; 3]) {
    let [_a] = arr;
      //^^^^ error: pattern requires 1 element but array has 3
}
"#,
        );
    }
}
