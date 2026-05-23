use stdx::format_to;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: record-pat-missing-fields
//
// This diagnostic is triggered if a record pattern omits fields without `..`.
pub(crate) fn record_pat_missing_fields(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::RecordPatMissingFields,
) -> Diagnostic {
    let mut message = String::from("missing structure fields:\n");
    for field in &d.missed_fields {
        format_to!(message, "- {}\n", field.display(ctx.sema.db, ctx.edition));
    }

    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0063"),
        message,
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn record_pat_missing_fields() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
fn baz(s: S) {
    let S { foo: _ } = s;
            //^ error: missing structure fields:
      //| - bar
}
"#,
        );
    }
}
