//! Provides a way to attach fixes to the diagnostics.
//! The same module also has all curret custom fixes for the diagnostics implemented.
mod change_case;

use hir::{diagnostics::Diagnostic, Semantics};
use ide_assists::AssistResolveStrategy;
use ide_db::RootDatabase;

use crate::Assist;

/// A [Diagnostic] that potentially has some fixes available.
///
/// [Diagnostic]: hir::diagnostics::Diagnostic
pub(crate) trait DiagnosticWithFixes: Diagnostic {
    /// `resolve` determines if the diagnostic should fill in the `edit` field
    /// of the assist.
    ///
    /// If `resolve` is false, the edit will be computed later, on demand, and
    /// can be omitted.
    fn fixes(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Vec<Assist>>;
}
