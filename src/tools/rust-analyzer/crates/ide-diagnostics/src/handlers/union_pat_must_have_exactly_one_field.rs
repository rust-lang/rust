use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: union-pat-must-have-exactly-one-field
//
// A union pattern does not have exactly one field.
pub(crate) fn union_pat_must_have_exactly_one_field(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::UnionPatMustHaveExactlyOneField,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0784"),
        "union patterns should have exactly one field",
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn union_pat_must_have_exactly_one_field() {
        check_diagnostics(
            r#"
union Bird {
    pigeon: u8,
    turtledove: u16,
}

fn main(bird: Bird) {
    unsafe {
        let Bird {} = bird;
            // ^^^^^^^ error: union patterns should have exactly one field
        let Bird { pigeon: 0, turtledove: 1 } = bird;
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: union patterns should have exactly one field
    }
}
"#,
        );
    }
}
