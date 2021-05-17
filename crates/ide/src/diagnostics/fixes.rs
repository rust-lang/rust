//! Provides a way to attach fixes to the diagnostics.
//! The same module also has all curret custom fixes for the diagnostics implemented.
mod change_case;
mod create_field;
mod fill_missing_fields;
mod remove_semicolon;
mod replace_with_find_map;
mod unresolved_module;
mod wrap_tail_expr;

use hir::{diagnostics::Diagnostic, Semantics};
use ide_assists::AssistResolveStrategy;
use ide_db::RootDatabase;

use crate::Assist;

/// A [Diagnostic] that potentially has a fix available.
///
/// [Diagnostic]: hir::diagnostics::Diagnostic
pub(crate) trait DiagnosticWithFix: Diagnostic {
    /// `resolve` determines if the diagnostic should fill in the `edit` field
    /// of the assist.
    ///
    /// If `resolve` is false, the edit will be computed later, on demand, and
    /// can be omitted.
    fn fix(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Assist>;
}
