//! Diagnostics emitted during body lowering.

use hir_expand::diagnostics::DiagnosticSink;

use crate::diagnostics::InactiveCode;

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum BodyDiagnostic {
    InactiveCode(InactiveCode),
}

impl BodyDiagnostic {
    pub(crate) fn add_to(&self, sink: &mut DiagnosticSink<'_>) {
        match self {
            BodyDiagnostic::InactiveCode(diag) => {
                sink.push(diag.clone());
            }
        }
    }
}
