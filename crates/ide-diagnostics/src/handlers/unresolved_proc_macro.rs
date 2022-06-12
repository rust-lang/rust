use crate::{Diagnostic, DiagnosticsContext, Severity};

// Diagnostic: unresolved-proc-macro
//
// This diagnostic is shown when a procedural macro can not be found. This usually means that
// procedural macro support is simply disabled (and hence is only a weak hint instead of an error),
// but can also indicate project setup problems.
//
// If you are seeing a lot of "proc macro not expanded" warnings, you can add this option to the
// `rust-analyzer.diagnostics.disabled` list to prevent them from showing. Alternatively you can
// enable support for procedural macros (see `rust-analyzer.procMacro.attributes.enable`).
pub(crate) fn unresolved_proc_macro(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedProcMacro,
    attr_proc_macros_enabled: bool,
) -> Diagnostic {
    // Use more accurate position if available.
    let display_range = d
        .precise_location
        .unwrap_or_else(|| ctx.sema.diagnostics_display_range(d.node.clone()).range);
    let message = match &d.macro_name {
        Some(name) => format!("proc macro `{}` not expanded", name),
        None => "proc macro not expanded".to_string(),
    };
    let message = format!(
        "{message}{}",
        if attr_proc_macros_enabled { "" } else { " (attribute macro expansion is disabled)" }
    );

    Diagnostic::new("unresolved-proc-macro", message, display_range)
        .severity(if attr_proc_macros_enabled { Severity::Error } else { Severity::WeakWarning })
}
