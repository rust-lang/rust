use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: functional-record-update-on-non-struct
//
// This diagnostic is triggered when functional record update syntax is used on
// something other than a struct.
pub(crate) fn functional_record_update_on_non_struct(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::FunctionalRecordUpdateOnNonStruct,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0436"),
        "functional record update syntax requires a struct",
        d.base_expr.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn enum_variant_record_update() {
        check_diagnostics(
            r#"
enum E {
    V { x: i32, y: i32 },
}

fn f(e: E) {
    let _ = E::V { x: 0, ..e };
                         //^ error: functional record update syntax requires a struct
}
"#,
        );
    }

    #[test]
    fn struct_record_update() {
        check_diagnostics(
            r#"
struct S {
    x: i32,
    y: i32,
}

fn f(s: S) {
    let _ = S { x: 0, ..s };
}
"#,
        );
    }
}
