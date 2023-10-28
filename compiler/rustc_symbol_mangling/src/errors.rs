//! Errors emitted by symbol_mangling.

use rustc_errors::{ErrorGuaranteed, IntoDiagnostic};
use rustc_span::Span;
use std::fmt;

pub struct TestOutput {
    pub span: Span,
    pub kind: Kind,
    pub content: String,
}

// This diagnostic doesn't need translation because (a) it doesn't contain any
// natural language, and (b) it's only used in tests. So we construct it
// manually and avoid the fluent machinery.
impl IntoDiagnostic<'_> for TestOutput {
    fn into_diagnostic(
        self,
        handler: &'_ rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let TestOutput { span, kind, content } = self;

        #[allow(rustc::untranslatable_diagnostic)]
        let mut diag = handler.struct_err(format!("{kind}({content})"));
        diag.set_span(span);
        diag
    }
}

pub enum Kind {
    SymbolName,
    Demangling,
    DemanglingAlt,
    DefPath,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::SymbolName => write!(f, "symbol-name"),
            Kind::Demangling => write!(f, "demangling"),
            Kind::DemanglingAlt => write!(f, "demangling-alt"),
            Kind::DefPath => write!(f, "def-path"),
        }
    }
}
