//! Errors emitted by ast_passes.

use rustc_errors::{fluent, AddSubdiagnostic, Applicability, Diagnostic};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_span::{Span, Symbol};

use crate::ast_validation::ForbiddenLetReason;

#[derive(SessionDiagnostic)]
#[diag(ast_passes::forbidden_let)]
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
#[diag(ast_passes::forbidden_let_stable)]
#[note]
pub struct ForbiddenLetStable {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::forbidden_assoc_constraint)]
pub struct ForbiddenAssocConstraint {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::keyword_lifetime)]
pub struct KeywordLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::invalid_label)]
pub struct InvalidLabel {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::invalid_visibility, code = "E0449")]
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
#[diag(ast_passes::trait_fn_const, code = "E0379")]
pub struct TraitFnConst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::forbidden_lifetime_bound)]
pub struct ForbiddenLifetimeBound {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::forbidden_non_lifetime_param)]
pub struct ForbiddenNonLifetimeParam {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::fn_param_too_many)]
pub struct FnParamTooMany {
    #[primary_span]
    pub span: Span,
    pub max_num_args: usize,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::fn_param_c_var_args_only)]
pub struct FnParamCVarArgsOnly {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::fn_param_c_var_args_not_last)]
pub struct FnParamCVarArgsNotLast {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::fn_param_doc_comment)]
pub struct FnParamDocComment {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::fn_param_forbidden_attr)]
pub struct FnParamForbiddenAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::fn_param_forbidden_self)]
#[note]
pub struct FnParamForbiddenSelf {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::forbidden_default)]
pub struct ForbiddenDefault {
    #[primary_span]
    pub span: Span,
    #[label]
    pub def_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::assoc_const_without_body)]
pub struct AssocConstWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::assoc_fn_without_body)]
pub struct AssocFnWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " {{ <body> }}", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::assoc_type_without_body)]
pub struct AssocTypeWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <type>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::const_without_body)]
pub struct ConstWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::static_without_body)]
pub struct StaticWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::ty_alias_without_body)]
pub struct TyAliasWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <type>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::fn_without_body)]
pub struct FnWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " {{ <body> }}", applicability = "has-placeholders")]
    pub replace_span: Span,
    #[subdiagnostic]
    pub extern_block_suggestion: Option<ExternBlockSuggestion>,
}

pub struct ExternBlockSuggestion {
    pub start_span: Span,
    pub end_span: Span,
    pub abi: Option<Symbol>,
}

impl AddSubdiagnostic for ExternBlockSuggestion {
    fn add_to_diagnostic(self, diag: &mut Diagnostic) {
        let start_suggestion = if let Some(abi) = self.abi {
            format!("extern \"{}\" {{", abi)
        } else {
            "extern {".to_owned()
        };
        let end_suggestion = " }".to_owned();

        diag.multipart_suggestion(
            fluent::ast_passes::extern_block_suggestion,
            vec![(self.start_span, start_suggestion), (self.end_span, end_suggestion)],
            Applicability::MaybeIncorrect,
        );
    }
}
