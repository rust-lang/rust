use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: union-expr-must-have-exactly-one-field
//
// A union expression does not have exactly one field.
pub(crate) fn union_expr_must_have_exactly_one_field(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::UnionExprMustHaveExactlyOneField,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0784"),
        "union expressions should have exactly one field",
        d.expr.map(|it| it.into()),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn union_expr_must_have_exactly_one_field() {
        check_diagnostics(
            r#"
union Bird {
    pigeon: u8,
    turtledove: u16,
}

fn main() {
    let bird = Bird { pigeon: 0 };
    let bird = Bird {};
            // ^^^^^^^ error: union expressions should have exactly one field
    let bird = Bird { pigeon: 0, turtledove: 1 };
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: union expressions should have exactly one field
}
"#,
        );
    }
}
