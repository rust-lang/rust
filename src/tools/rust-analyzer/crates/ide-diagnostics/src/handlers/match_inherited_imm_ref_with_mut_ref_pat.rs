use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: match-inherited-imm-ref-with-mut-ref-pat
//
// Triggered when a `&mut` pattern attempts to consume an
// inherited shared (`&`) reference.
pub(crate) fn match_inherited_imm_ref_with_mut_ref_pat(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::MatchInheritedImmRefWithMutRefPat,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0308"),
        "cannot match inherited `&` with `&mut` pattern",
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn eat_outer_inherited_imm_ref_with_mut_pat() {
        check_diagnostics(
            r#"
#![feature(ref_pat_eat_one_layer_2024)]

fn main() {
    let [&mut x] = &[0];
       //^^^^^^ error: cannot match inherited `&` with `&mut` pattern
    let [&mut x] = &mut [0];
    let [&x] = &[0];
    let [&x] = &mut [0];
}
"#,
        );
    }
}
