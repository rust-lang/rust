//! Errors emitted by ast_passes.

use rustc_errors::fluent;
use rustc_errors::{AddSubdiagnostic, Diagnostic};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_span::{Span, Symbol};

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
            }
        }
    }
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::forbidden_assoc_constraint)]
pub struct ForbiddenAssocConstraint {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::keyword_lifetime)]
pub struct KeywordLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::invalid_label)]
pub struct InvalidLabel {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::invalid_visibility, code = "E0449")]
pub struct InvalidVisibility {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes::implied)]
    pub implied: Option<Span>,
    #[subdiagnostic]
    pub note: Option<InvalidVisibilityNote>,
}

#[derive(SessionSubdiagnostic)]
pub enum InvalidVisibilityNote {
    #[note(ast_passes::individual_impl_items)]
    IndividualImplItems,
    #[note(ast_passes::individual_foreign_items)]
    IndividualForeignItems,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::trait_fn_async, code = "E0706")]
#[note]
#[note(ast_passes::note2)]
pub struct TraitFnAsync {
    #[primary_span]
    pub fn_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::trait_fn_const, code = "E0379")]
pub struct TraitFnConst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::forbidden_lifetime_bound)]
pub struct ForbiddenLifetimeBound {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::forbidden_non_lifetime_param)]
pub struct ForbiddenNonLifetimeParam {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::too_many_params)]
pub struct TooManyParams {
    #[primary_span]
    pub span: Span,
    pub max_num_args: usize,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::c_var_args_is_sole_param)]
pub struct CVarArgsIsSoleParam {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::c_var_args_not_last)]
pub struct CVarArgsNotLast {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::doc_comment_on_fn_param)]
pub struct DocCommentOnFnParam {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(ast_passes::forbidden_attr_on_fn_param)]
pub struct ForbiddenAttrOnFnParam {
    #[primary_span]
    pub span: Span,
}
