use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, Severity};

// Diagnostic: unimplemented-builtin-macro
//
// This diagnostic is shown for builtin macros which are not yet implemented by rust-analyzer
pub(crate) fn unimplemented_builtin_macro(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnimplementedBuiltinMacro,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::Ra("unimplemented-builtin-macro", Severity::WeakWarning),
        "unimplemented built-in macro".to_owned(),
        d.node,
    )
    .stable()
}
