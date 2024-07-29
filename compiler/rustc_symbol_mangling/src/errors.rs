//! Errors emitted by symbol_mangling.

use std::fmt;

use rustc_errors::{Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level};
use rustc_span::Span;

pub struct TestOutput {
    pub span: Span,
    pub kind: Kind,
    pub content: String,
}

// This diagnostic doesn't need translation because (a) it doesn't contain any
// natural language, and (b) it's only used in tests. So we construct it
// manually and avoid the fluent machinery.
impl<G: EmissionGuarantee> Diagnostic<'_, G> for TestOutput {
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let TestOutput { span, kind, content } = self;

        #[allow(rustc::untranslatable_diagnostic)]
        Diag::new(dcx, level, format!("{kind}({content})")).with_span(span)
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
