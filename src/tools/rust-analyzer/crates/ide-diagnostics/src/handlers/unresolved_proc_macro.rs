use hir::db::DefDatabase;

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
    proc_macros_enabled: bool,
    proc_attr_macros_enabled: bool,
) -> Diagnostic {
    // Use more accurate position if available.
    let display_range = ctx.resolve_precise_location(&d.node, d.precise_location);

    let config_enabled = match d.kind {
        hir::MacroKind::Attr => proc_macros_enabled && proc_attr_macros_enabled,
        _ => proc_macros_enabled,
    };

    let message = match &d.macro_name {
        Some(name) => format!("proc macro `{name}` not expanded"),
        None => "proc macro not expanded".to_string(),
    };
    let severity = if config_enabled { Severity::Error } else { Severity::WeakWarning };
    let def_map = ctx.sema.db.crate_def_map(d.krate);
    let message = format!(
        "{message}: {}",
        if config_enabled {
            def_map.proc_macro_loading_error().unwrap_or("proc macro not found in the built dylib")
        } else {
            match d.kind {
                hir::MacroKind::Attr if proc_macros_enabled => {
                    "attribute macro expansion is disabled"
                }
                _ => "proc-macro expansion is disabled",
            }
        },
    );

    Diagnostic::new("unresolved-proc-macro", message, display_range).severity(severity)
}
