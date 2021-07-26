use hir::{db::AstDatabase, InFile};
use syntax::{AstNode, SyntaxNodePtr};

use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: unresolved-macro-call
//
// This diagnostic is triggered if rust-analyzer is unable to resolve the path
// to a macro in a macro invocation.
pub(crate) fn unresolved_macro_call(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedMacroCall,
) -> Diagnostic {
    let last_path_segment = ctx.sema.db.parse_or_expand(d.macro_call.file_id).and_then(|root| {
        d.macro_call
            .value
            .to_node(&root)
            .path()
            .and_then(|it| it.segment())
            .and_then(|it| it.name_ref())
            .map(|it| InFile::new(d.macro_call.file_id, SyntaxNodePtr::new(it.syntax())))
    });
    let diagnostics = last_path_segment.unwrap_or_else(|| d.macro_call.clone().map(|it| it.into()));

    Diagnostic::new(
        "unresolved-macro-call",
        format!("unresolved macro `{}!`", d.path),
        ctx.sema.diagnostics_display_range(diagnostics).range,
    )
    .experimental()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn unresolved_macro_diag() {
        check_diagnostics(
            r#"
fn f() {
    m!();
} //^ error: unresolved macro `m!`

"#,
        );
    }

    #[test]
    fn test_unresolved_macro_range() {
        check_diagnostics(
            r#"
foo::bar!(92);
   //^^^ error: unresolved macro `foo::bar!`
"#,
        );
    }

    #[test]
    fn unresolved_legacy_scope_macro() {
        check_diagnostics(
            r#"
macro_rules! m { () => {} }

m!(); m2!();
    //^^ error: unresolved macro `m2!`
"#,
        );
    }

    #[test]
    fn unresolved_module_scope_macro() {
        check_diagnostics(
            r#"
mod mac {
#[macro_export]
macro_rules! m { () => {} } }

self::m!(); self::m2!();
                //^^ error: unresolved macro `self::m2!`
"#,
        );
    }
}
