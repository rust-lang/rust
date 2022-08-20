use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic)]
#[error(lint::malformed_attribute, code = "E0452")]
pub struct MalformedAttribute {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: MalformedAttributeSub,
}

#[derive(SessionSubdiagnostic)]
pub enum MalformedAttributeSub {
    #[label(lint::bad_attribute_argument)]
    BadAttributeArgument(#[primary_span] Span),
    #[label(lint::reason_must_be_string_literal)]
    ReasonMustBeStringLiteral(#[primary_span] Span),
    #[label(lint::reason_must_come_last)]
    ReasonMustComeLast(#[primary_span] Span),
}

#[derive(SessionDiagnostic)]
#[error(lint::unknown_tool, code = "E0710")]
pub struct UnknownTool {
    #[primary_span]
    pub span: Option<Span>,
    pub tool_name: Symbol,
    pub lint_name: String,
    #[help]
    pub is_nightly_build: Option<()>,
}
