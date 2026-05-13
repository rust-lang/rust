use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: expected-array-or-slice-pat
//
// This diagnostic is triggered when an array or slice pattern is matched
// against a type that is neither an array nor a slice.
pub(crate) fn expected_array_or_slice_pat(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::ExpectedArrayOrSlicePat<'_>,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0529"),
        format!(
            "expected an array or slice, found {}",
            d.found.display(ctx.sema.db, ctx.display_target)
        ),
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn expected_array_or_slice() {
        check_diagnostics(
            r#"
fn f([_a, _b]: i32) {}
   //^^^^^^^^ error: expected an array or slice, found i32
"#,
        );
    }

    #[test]
    fn expected_array_or_slice_let_pattern() {
        check_diagnostics(
            r#"
fn f(x: i32) {
    let [_a, _b] = x;
      //^^^^^^^^ error: expected an array or slice, found i32
}
"#,
        );
    }
}
