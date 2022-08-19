use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[error(lint::unknown_tool, code = "E0710")]
pub struct UnknownTool {
    #[primary_span]
    pub span: Option<Span>,
    pub tool_name: String,
    pub lint_name: String,
    #[help]
    pub is_nightly_build: Option<()>,
}
