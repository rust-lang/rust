use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unresolved-associated-type
//
// This diagnostic is triggered when an associated type is not found for a type parameter or `Self`.
pub(crate) fn unresolved_associated_type(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::UnresolvedAssociatedType,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0220"),
        "associated type not found",
        d.segment.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn unresolved_associated_type_generic_param() {
        check_diagnostics(
            r#"
trait Trait {}
fn example<T: Trait>(val: T::Item) {}
                           //^^^^ error: associated type not found
"#,
        );
    }

    #[test]
    fn unresolved_associated_type_self() {
        check_diagnostics(
            r#"
trait Trait {
    fn example();
}
struct S;
impl Trait for S {
    fn example() {
        let _val: Self::Item;
                      //^^^^ error: associated type not found
    }
}
"#,
        );
    }
}
