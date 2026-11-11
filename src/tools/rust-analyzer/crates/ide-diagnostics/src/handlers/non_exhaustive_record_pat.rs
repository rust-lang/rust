use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: non-exhaustive-record-pat
//
// This diagnostic is triggered if a record pattern destructures a `#[non_exhaustive]`
// struct or enum variant from another crate without `..`.
pub(crate) fn non_exhaustive_record_pat(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::NonExhaustiveRecordPat,
) -> Diagnostic {
    let item = match d.variant {
        hir::Variant::Struct(_) => "struct",
        hir::Variant::Union(_) => "union",
        hir::Variant::EnumVariant(_) => "variant",
    };

    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0638"),
        format!("`..` required with {item} marked as non-exhaustive"),
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn reports_external_non_exhaustive_struct_pattern_without_rest() {
        check_diagnostics(
            r#"
//- /lib.rs crate:lib
#[non_exhaustive]
pub struct S {
    pub field: u32,
}

fn local_ok(s: S) {
    let S { field } = s;
    let _ = field;
}

//- /main.rs crate:main deps:lib
fn main(s: lib::S) {
    let lib::S { field } = s;
      //^^^^^^^^^^^^^^^^ error: `..` required with struct marked as non-exhaustive
    let _ = field;
}
"#,
        );
    }

    #[test]
    fn reports_external_non_exhaustive_variant_pattern_without_rest() {
        check_diagnostics(
            r#"
//- /lib.rs crate:lib
pub enum E {
    #[non_exhaustive]
    V { field: u32 },
}

fn local_ok(e: E) {
    let E::V { field } = e;
    let _ = field;
}

//- /main.rs crate:main deps:lib
fn main(e: lib::E) {
    let lib::E::V { field } = e;
      //^^^^^^^^^^^^^^^^^^^ error: `..` required with variant marked as non-exhaustive
    let _ = field;
}
"#,
        );
    }
}
