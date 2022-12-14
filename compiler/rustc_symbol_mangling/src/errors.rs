//! Errors emitted by symbol_mangling.

use rustc_errors::{DiagnosticArgValue, IntoDiagnosticArg};
use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(symbol_mangling_test_output)]
pub struct TestOutput {
    #[primary_span]
    pub span: Span,
    pub kind: Kind,
    pub content: String,
}

pub enum Kind {
    SymbolName,
    Demangling,
    DemanglingAlt,
    DefPath,
}

impl IntoDiagnosticArg for Kind {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        let kind = match self {
            Kind::SymbolName => "symbol-name",
            Kind::Demangling => "demangling",
            Kind::DemanglingAlt => "demangling-alt",
            Kind::DefPath => "def-path",
        }
        .into();
        DiagnosticArgValue::Str(kind)
    }
}
