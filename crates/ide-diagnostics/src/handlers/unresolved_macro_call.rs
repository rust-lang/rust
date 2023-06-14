use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unresolved-macro-call
//
// This diagnostic is triggered if rust-analyzer is unable to resolve the path
// to a macro in a macro invocation.
pub(crate) fn unresolved_macro_call(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedMacroCall,
) -> Diagnostic {
    // Use more accurate position if available.
    let display_range = ctx.resolve_precise_location(&d.macro_call, d.precise_location);
    let bang = if d.is_bang { "!" } else { "" };
    Diagnostic::new(
        DiagnosticCode::RustcHardError("unresolved-macro-call"),
        format!("unresolved macro `{}{bang}`", d.path.display(ctx.sema.db)),
        display_range,
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
