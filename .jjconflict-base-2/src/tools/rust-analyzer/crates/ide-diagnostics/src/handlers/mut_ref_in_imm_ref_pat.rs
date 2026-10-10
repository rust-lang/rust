use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: mut-ref-in-imm-ref-pat
//
// This diagnostic is triggered when a binding tries to mutably borrow through
// an `&` pattern.
pub(crate) fn mut_ref_in_imm_ref_pat(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::MutRefInImmRefPat,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0596"),
        "cannot borrow as mutable inside an `&` pattern",
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn mut_ref_in_imm_ref_pat() {
        check_diagnostics(
            r#"
#![feature(ref_pat_eat_one_layer_2024)]

fn main() {
    let &ref mut _x = &mut 0;
       //^^^^^^^^^^ error: cannot borrow as mutable inside an `&` pattern
}
"#,
        );
    }
}
