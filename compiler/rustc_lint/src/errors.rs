use rustc_errors::codes::*;
use rustc_errors::{Diag, EmissionGuarantee, Subdiagnostic, msg};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_session::lint::Level;
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag("{$lint_level}({$lint_source}) incompatible with previous forbid", code = E0453)]
pub(crate) struct OverruledAttribute<'a> {
    #[primary_span]
    pub span: Span,
    #[label("overruled by previous forbid")]
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
                diag.note(msg!("`forbid` lint level is the default for {$id}"));
                diag.arg("id", id);
            }
            OverruledAttributeSub::NodeSource { span, reason } => {
                diag.span_label(span, msg!("`forbid` level set here"));
                if let Some(rationale) = reason {
                    diag.note(rationale.to_string());
                }
            }
            OverruledAttributeSub::CommandLineSource => {
                diag.note(msg!("`forbid` lint level was set on command line"));
            }
        }
    }
}

#[derive(Diagnostic)]
#[diag("malformed lint attribute input", code = E0452)]
pub(crate) struct MalformedAttribute {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: MalformedAttributeSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum MalformedAttributeSub {
    #[label("bad attribute argument")]
    BadAttributeArgument(#[primary_span] Span),
    #[label("reason must be a string literal")]
    ReasonMustBeStringLiteral(#[primary_span] Span),
    #[label("reason in lint attribute must come last")]
    ReasonMustComeLast(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("unknown tool name `{$tool_name}` found in scoped lint: `{$tool_name}::{$lint_name}`", code = E0710)]
pub(crate) struct UnknownToolInScopedLint {
    #[primary_span]
    pub span: Option<Span>,
    pub tool_name: Symbol,
    pub lint_name: String,
    #[help("add `#![register_tool({$tool_name})]` to the crate root")]
    pub is_nightly_build: bool,
}

#[derive(Diagnostic)]
#[diag("`...` range patterns are deprecated", code = E0783)]
pub(crate) struct BuiltinEllipsisInclusiveRangePatterns {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use `..=` for an inclusive range",
        style = "short",
        code = "{replace}",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
    pub replace: String,
}

#[derive(Subdiagnostic)]
#[note("requested on the command line with `{$level} {$lint_name}`")]
pub(crate) struct RequestedLevel<'a> {
    pub level: Level,
    pub lint_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("`{$lint_group}` lint group is not supported with ´--force-warn´", code = E0602)]
pub(crate) struct UnsupportedGroup {
    pub lint_group: String,
}

#[derive(Diagnostic)]
#[diag("unknown lint tool: `{$tool_name}`", code = E0602)]
pub(crate) struct CheckNameUnknownTool<'a> {
    pub tool_name: Symbol,
    #[subdiagnostic]
    pub sub: RequestedLevel<'a>,
}
