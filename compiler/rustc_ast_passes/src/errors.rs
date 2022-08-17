//! Errors emitted by ast_passes.

use rustc_errors::fluent;
use rustc_errors::{AddSubdiagnostic, Diagnostic};
use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

use crate::ast_validation::ForbiddenLetReason;

#[derive(SessionDiagnostic)]
#[error(ast_passes::forbidden_let)]
#[note]
pub struct ForbiddenLet {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub(crate) reason: ForbiddenLetReason,
}

impl AddSubdiagnostic for ForbiddenLetReason {
    fn add_to_diagnostic(self, diag: &mut Diagnostic) {
        match self {
            Self::GenericForbidden => {}
            Self::NotSupportedOr(span) => {
                diag.span_note(span, fluent::ast_passes::not_supported_or);
            }
            Self::NotSupportedParentheses(span) => {
                diag.span_note(span, fluent::ast_passes::not_supported_parentheses);
            },
        }
    }
}
