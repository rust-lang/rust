use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: generic-default-refers-to-self
//
// This diagnostic is shown when a generic default refers to `Self`
pub(crate) fn generic_default_refers_to_self(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::GenericDefaultRefersToSelf,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0735"),
        "generic parameters cannot use `Self` in their defaults",
        d.segment.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn plain_self() {
        check_diagnostics(
            r#"
struct Foo<T = Self>(T);
            // ^^^^ error: generic parameters cannot use `Self` in their defaults
"#,
        );
    }

    #[test]
    fn self_as_generic() {
        check_diagnostics(
            r#"
struct Wrapper<T>(T);
struct Foo<T = Wrapper<Self>>(T);
                    // ^^^^ error: generic parameters cannot use `Self` in their defaults
"#,
        );
    }
}
