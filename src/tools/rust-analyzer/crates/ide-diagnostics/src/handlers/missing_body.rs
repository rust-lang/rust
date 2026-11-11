use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: missing-body
//
// This diagnostic is triggered when a body is missing.
pub(crate) fn missing_body(ctx: &DiagnosticsContext<'_, '_>, d: &hir::MissingBody) -> Diagnostic {
    let message = match d.kind {
        hir::MissingBodyItemKind::AssocConst => "associated constant in `impl` without body",
        hir::MissingBodyItemKind::AssocType => "associated type in `impl` without body",
        hir::MissingBodyItemKind::Const => "free constant item without body",
        hir::MissingBodyItemKind::Static => "free static item without body",
        hir::MissingBodyItemKind::TypeAlias => "free type alias without body",
    };
    Diagnostic::new_with_syntax_node_ptr(ctx, DiagnosticCode::SyntaxError, message, d.node).stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn associated_const() {
        check_diagnostics(
            r#"
trait Foo { const BAR: u32; }
impl Foo for () { const BAR: u32; }
                //^^^^^^^^^^^^^^^ error: associated constant in `impl` without body
        "#,
        );
    }

    #[test]
    fn associated_type_impl() {
        check_diagnostics(
            r#"
trait Foo { type Bar; }
impl Foo for () { type Bar; }
                //^^^^^^^^^ error: associated type in `impl` without body
        "#,
        );
    }

    #[test]
    fn free_const() {
        check_diagnostics(
            r#"
  const FOO: u32;
//^^^^^^^^^^^^^^^ error: free constant item without body
        "#,
        );
    }

    #[test]
    fn free_static() {
        check_diagnostics(
            r#"
  static FOO: u32;
//^^^^^^^^^^^^^^^^ error: free static item without body
        "#,
        );
    }

    #[test]
    fn type_alias_module() {
        check_diagnostics(
            r#"
  type Foo;
//^^^^^^^^^ error: free type alias without body
        "#,
        );
    }
}
