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
    CommandLineSource { id: Symbol },
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
            OverruledAttributeSub::CommandLineSource { id } => {
                diag.note(msg!("`forbid` lint level was set on command line (`-F {$id}`)"));
                diag.arg("id", id);
            }
        }
    }
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
