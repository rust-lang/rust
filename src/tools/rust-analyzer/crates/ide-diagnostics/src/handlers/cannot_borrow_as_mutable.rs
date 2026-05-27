use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: cannot-borrow-as-mutable
//
// This diagnostic is triggered when a binding tries to mutably borrow through
// an `&` pattern.
pub(crate) fn cannot_borrow_as_mutable(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::CannotBorrowAsMutable,
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
    fn cannot_borrow_as_mutable_inside_shared_ref_pattern() {
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
