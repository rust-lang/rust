use rustc_errors::{fluent, AddSubdiagnostic};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic)]
#[error(lint::overruled_attribute, code = "E0453")]
pub struct OverruledAttribute {
    #[primary_span]
    pub span: Span,
    #[label]
    pub overruled: Span,
    pub lint_level: String,
    pub lint_source: Symbol,
    #[subdiagnostic]
    pub sub: OverruledAttributeSub,
}
//
pub enum OverruledAttributeSub {
    DefaultSource { id: String },
    NodeSource { span: Span, reason: Option<Symbol> },
    CommandLineSource,
}

impl AddSubdiagnostic for OverruledAttributeSub {
    fn add_to_diagnostic(self, diag: &mut rustc_errors::Diagnostic) {
        match self {
            OverruledAttributeSub::DefaultSource { id } => {
                diag.note(fluent::lint::default_source);
                diag.set_arg("id", id);
            }
            OverruledAttributeSub::NodeSource { span, reason } => {
                diag.span_label(span, fluent::lint::node_source);
                if let Some(rationale) = reason {
                    diag.note(rationale.as_str());
                }
            }
            OverruledAttributeSub::CommandLineSource => {
                diag.note(fluent::lint::command_line_source);
            }
        }
    }
}

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

#[derive(SessionDiagnostic)]
#[error(lint::builtin_ellipsis_inclusive_range_patterns, code = "E0783")]
pub struct BuiltinEllpisisInclusiveRangePatterns {
    #[primary_span]
    pub span: Span,
    #[suggestion_short(code = "{replace}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub replace: String,
}
