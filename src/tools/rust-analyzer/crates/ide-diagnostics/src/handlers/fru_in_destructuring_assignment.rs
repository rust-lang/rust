use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: fru-in-destructuring-assignment
//
// This diagnostic is triggered when a destructuring assignment contains functional record update
pub(crate) fn fru_in_destructuring_assignment(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::FruInDestructuringAssignment,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::SyntaxError,
        "functional record updates are not allowed in destructuring assignments",
        d.node.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_diagnostics_with_disabled};

    #[test]
    fn spread_variable() {
        check_diagnostics_with_disabled(
            r#"
struct Foo { bar: u32, baz: u32 }
fn test(f: Foo, g: Foo, mut bar: u32, mut baz: u32) {
    Foo { ..g } = f;
         // ^ error: functional record updates are not allowed in destructuring assignments
    Foo { bar, ..g } = f;
              // ^ error: functional record updates are not allowed in destructuring assignments
    Foo { bar, baz, ..g } = f;
                   // ^ error: functional record updates are not allowed in destructuring assignments
}
        "#,
            // We don't end up using neither `bar` nor `baz`
            &["unused_variables"],
        );
    }

    #[test]
    fn spread_default() {
        check_diagnostics(
            r#"
struct Foo { bar: u32, baz: u32 }
fn test(f: Foo) {
    Foo { ..Default::default() } = f;
         // ^^^^^^^^^^^^^^^^^^ error: functional record updates are not allowed in destructuring assignments
}
        "#,
        );
    }

    #[test]
    fn spread_struct() {
        check_diagnostics(
            r#"
struct Foo { bar: u32, baz: u32 }
fn test(f: Foo) {
    Foo { ..Foo { bar: 0, baz: 0 } } = f;
         // ^^^^^^^^^^^^^^^^^^^^^^ error: functional record updates are not allowed in destructuring assignments
}
        "#,
        );
    }
}
