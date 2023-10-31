use crate::fluent_generated as fluent;
use rustc_errors::{
    AddToDiagnostic, Diagnostic, ErrorGuaranteed, Handler, IntoDiagnostic, SubdiagnosticMessage,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_session::lint::Level;
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag(lint_overruled_attribute, code = "E0453")]
pub struct OverruledAttribute<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub overruled: Span,
    pub lint_level: &'a str,
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

impl AddToDiagnostic for OverruledAttributeSub {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        match self {
            OverruledAttributeSub::DefaultSource { id } => {
                diag.note(fluent::lint_default_source);
                diag.set_arg("id", id);
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
#[diag(lint_malformed_attribute, code = "E0452")]
pub struct MalformedAttribute {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: MalformedAttributeSub,
}

#[derive(Subdiagnostic)]
pub enum MalformedAttributeSub {
    #[label(lint_bad_attribute_argument)]
    BadAttributeArgument(#[primary_span] Span),
    #[label(lint_reason_must_be_string_literal)]
    ReasonMustBeStringLiteral(#[primary_span] Span),
    #[label(lint_reason_must_come_last)]
    ReasonMustComeLast(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(lint_unknown_tool_in_scoped_lint, code = "E0710")]
pub struct UnknownToolInScopedLint {
    #[primary_span]
    pub span: Option<Span>,
    pub tool_name: Symbol,
    pub lint_name: String,
    #[help]
    pub is_nightly_build: Option<()>,
}

#[derive(Diagnostic)]
#[diag(lint_builtin_ellipsis_inclusive_range_patterns, code = "E0783")]
pub struct BuiltinEllipsisInclusiveRangePatterns {
    #[primary_span]
    pub span: Span,
    #[suggestion(style = "short", code = "{replace}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub replace: String,
}

#[derive(Subdiagnostic)]
#[note(lint_requested_level)]
pub struct RequestedLevel {
    pub level: Level,
    pub lint_name: String,
}

#[derive(Diagnostic)]
#[diag(lint_unsupported_group, code = "E0602")]
pub struct UnsupportedGroup {
    pub lint_group: String,
}

pub struct CheckNameUnknown {
    pub lint_name: String,
    pub suggestion: Option<Symbol>,
    pub sub: RequestedLevel,
}

impl IntoDiagnostic<'_> for CheckNameUnknown {
    fn into_diagnostic(
        self,
        handler: &Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::lint_check_name_unknown);
        diag.code(rustc_errors::error_code!(E0602));
        if let Some(suggestion) = self.suggestion {
            diag.help(fluent::lint_help);
            diag.set_arg("suggestion", suggestion);
        }
        diag.set_arg("lint_name", self.lint_name);
        diag.subdiagnostic(self.sub);
        diag
    }
}

#[derive(Diagnostic)]
#[diag(lint_check_name_unknown_tool, code = "E0602")]
pub struct CheckNameUnknownTool {
    pub tool_name: Symbol,
    #[subdiagnostic]
    pub sub: RequestedLevel,
}

#[derive(Diagnostic)]
#[diag(lint_check_name_warning)]
pub struct CheckNameWarning {
    pub msg: String,
    #[subdiagnostic]
    pub sub: RequestedLevel,
}

#[derive(Diagnostic)]
#[diag(lint_check_name_deprecated)]
pub struct CheckNameDeprecated {
    pub lint_name: String,
    pub new_name: String,
    #[subdiagnostic]
    pub sub: RequestedLevel,
}
