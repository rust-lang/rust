use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: non-exhaustive-record-expr
//
// This diagnostic is triggered if a struct expression constructs a `#[non_exhaustive]`
// struct from another crate.
pub(crate) fn non_exhaustive_record_expr(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::NonExhaustiveRecordExpr,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0639"),
        "cannot create non-exhaustive struct using struct expression",
        d.expr.map(|it| it.into()),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn reports_external_non_exhaustive_struct_literal() {
        check_diagnostics(
            r#"
//- /lib.rs crate:lib
#[non_exhaustive]
pub struct S {
    pub field: u32,
}

fn local_ok() {
    let _ = S { field: 0 };
}

//- /main.rs crate:main deps:lib
fn main() {
    let _ = lib::S { field: 0 };
          //^^^^^^^^^^^^^^^^^^^ error: cannot create non-exhaustive struct using struct expression
}
"#,
        );
    }
}
