use rustc_errors::codes::*;
use rustc_errors::{Diag, EmissionGuarantee, Subdiagnostic};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_session::lint::Level;
use rustc_span::{Span, Symbol};

use crate::fluent_generated as fluent;

#[derive(Diagnostic)]
#[diag(lint_overruled_attribute, code = E0453)]
pub(crate) struct OverruledAttribute<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub overruled: Span,
    pub lint_level: &'a str,
    pub lint_source: Symbol,
    #[subdiagnostic]
    pub sub: OverruledAttributeSub,
}

pub(crate) enum OverruledAttributeSub {
    DefaultSource { id: String },
    NodeSource { span: Span, reason: Option<Symbol> },
    CommandLineSource,
}

impl Subdiagnostic for OverruledAttributeSub {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        match self {
            OverruledAttributeSub::DefaultSource { id } => {
                diag.note(fluent::lint_default_source);
                diag.arg("id", id);
            }
            OverruledAttributeSub::NodeSource { span, reason } => {
                diag.span_label(span, fluent::lint_node_source);
                if let Some(rationale) = reason {
                    #[allow(rustc::untranslatable_diagnostic)]
                    diag.note(rationale.to_string());
                }
            }
            OverruledAttributeSub::CommandLineSource => {
                diag.note(fluent::lint_command_line_source);
            }
        }
    }
}

#[derive(Diagnostic)]
#[diag(lint_malformed_attribute, code = E0452)]
pub(crate) struct MalformedAttribute {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: MalformedAttributeSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum MalformedAttributeSub {
    #[label(lint_bad_attribute_argument)]
    BadAttributeArgument(#[primary_span] Span),
    #[label(lint_reason_must_be_string_literal)]
    ReasonMustBeStringLiteral(#[primary_span] Span),
    #[label(lint_reason_must_come_last)]
    ReasonMustComeLast(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(lint_unknown_tool_in_scoped_lint, code = E0710)]
pub(crate) struct UnknownToolInScopedLint {
    #[primary_span]
    pub span: Option<Span>,
    pub tool_name: Symbol,
    pub lint_name: String,
    #[help]
    pub is_nightly_build: bool,
}

#[derive(Diagnostic)]
#[diag(lint_builtin_ellipsis_inclusive_range_patterns, code = E0783)]
pub(crate) struct BuiltinEllipsisInclusiveRangePatterns {
    #[primary_span]
    pub span: Span,
    #[suggestion(style = "short", code = "{replace}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub replace: String,
}

#[derive(Subdiagnostic)]
#[note(lint_requested_level)]
pub(crate) struct RequestedLevel<'a> {
    pub level: Level,
    pub lint_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(lint_unsupported_group, code = E0602)]
pub(crate) struct UnsupportedGroup {
    pub lint_group: String,
}

#[derive(Diagnostic)]
#[diag(lint_check_name_unknown_tool, code = E0602)]
pub(crate) struct CheckNameUnknownTool<'a> {
    pub tool_name: Symbol,
    #[subdiagnostic]
    pub sub: RequestedLevel<'a>,
}
