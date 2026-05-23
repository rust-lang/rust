use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: union-pat-has-rest
//
// A union pattern uses `..`.
pub(crate) fn union_pat_has_rest(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::UnionPatHasRest,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0784"),
        "union patterns cannot use `..`",
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn union_pat_has_rest() {
        check_diagnostics(
            r#"
union Bird {
    pigeon: u8,
    turtledove: u16,
}

fn main(bird: Bird) {
    unsafe {
        let Bird { pigeon: 0, .. } = bird;
            // ^^^^^^^^^^^^^^^^^^^^^ error: union patterns cannot use `..`
    }
}
"#,
        );
    }
}
