use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: infer-vars-not-allowed
//
// This diagnostic is triggered when `_` is used where type
// inference is not allowed.
pub(crate) fn infer_vars_not_allowed(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::InferVarsNotAllowed,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0121"),
        "the type placeholder `_` is not allowed within types on item signatures",
        d.node,
    )
}
#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;
    #[test]
    fn type_alias() {
        check_diagnostics(
            r#"
type Foo = _;
        // ^ error: the type placeholder `_` is not allowed within types on item signatures
        "#,
        );
    }
    #[test]
    fn const_item() {
        check_diagnostics(
            r#"
const X: _ = 0;
      // ^ error: the type placeholder `_` is not allowed within types on item signatures
        "#,
        );
    }

    #[test]
    fn static_item() {
        check_diagnostics(
            r#"
static Y: _ = 0;
       // ^ error: the type placeholder `_` is not allowed within types on item signatures
        "#,
        );
    }
}
