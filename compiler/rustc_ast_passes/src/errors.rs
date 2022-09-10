//! Errors emitted by ast_passes.

use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

use crate::ast_validation::ForbiddenLetReason;

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_let)]
#[note]
pub struct ForbiddenLet {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub(crate) reason: ForbiddenLetReason,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_let_stable)]
#[note]
pub struct ForbiddenLetStable {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_assoc_constraint)]
pub struct ForbiddenAssocConstraint {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_keyword_lifetime)]
pub struct KeywordLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_label)]
pub struct InvalidLabel {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_visibility, code = "E0449")]
pub struct InvalidVisibility {
    #[primary_span]
    pub span: Span,
    #[label(implied)]
    pub implied: Option<Span>,
    #[subdiagnostic]
    pub note: Option<InvalidVisibilityNote>,
}

#[derive(Subdiagnostic)]
pub enum InvalidVisibilityNote {
    #[note(individual_impl_items)]
    IndividualImplItems,
    #[note(individual_foreign_items)]
    IndividualForeignItems,
}

#[derive(Diagnostic)]
#[diag(ast_passes_trait_fn_const, code = "E0379")]
pub struct TraitFnConst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_lifetime_bound)]
pub struct ForbiddenLifetimeBound {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_non_lifetime_param)]
pub struct ForbiddenNonLifetimeParam {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_too_many)]
pub struct FnParamTooMany {
    #[primary_span]
    pub span: Span,
    pub max_num_args: usize,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_c_var_args_only)]
pub struct FnParamCVarArgsOnly {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_c_var_args_not_last)]
pub struct FnParamCVarArgsNotLast {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_doc_comment)]
pub struct FnParamDocComment {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_forbidden_attr)]
pub struct FnParamForbiddenAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_forbidden_self)]
#[note]
pub struct FnParamForbiddenSelf {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_default)]
pub struct ForbiddenDefault {
    #[primary_span]
    pub span: Span,
    #[label]
    pub def_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_assoc_const_without_body)]
pub struct AssocConstWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_assoc_fn_without_body)]
pub struct AssocFnWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " {{ <body> }}", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_impl_assoc_ty_without_body)]
pub struct ImplAssocTyWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <type>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_without_body)]
pub struct ConstWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_static_without_body)]
pub struct StaticWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_ty_alias_without_body)]
pub struct TyAliasWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <type>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_without_body)]
pub struct FnWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " {{ <body> }}", applicability = "has-placeholders")]
    pub replace_span: Span,
    #[subdiagnostic]
    pub extern_block_suggestion: Option<ExternBlockSuggestion>,
}

#[derive(Subdiagnostic)]
pub enum ExternBlockSuggestion {
    #[multipart_suggestion(ast_passes_extern_block_suggestion, applicability = "maybe-incorrect")]
    Implicit {
        #[suggestion_part(code = "extern {{")]
        start_span: Span,
        #[suggestion_part(code = " }}")]
        end_span: Span,
    },
    #[multipart_suggestion(ast_passes_extern_block_suggestion, applicability = "maybe-incorrect")]
    Explicit {
        #[suggestion_part(code = "extern \"{abi}\" {{")]
        start_span: Span,
        #[suggestion_part(code = " }}")]
        end_span: Span,
        abi: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag(ast_passes_ty_alias_with_bound)]
pub struct TyAliasWithBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_foreign_ty_with_bound)]
pub struct ForeignTyWithBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_impl_assoc_ty_with_bound)]
pub struct ImplAssocTyWithBound {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::foreign_ty_with_generic_param)]
#[note(ast_passes::more_extern_note)]
pub struct ForeignTyWithGenericParam {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::foreign_ty_with_where_clause)]
#[note(ast_passes::more_extern_note)]
pub struct ForeignTyWithWhereClause {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::foreign_ty_with_body)]
#[note(ast_passes::more_extern_note)]
pub struct ForeignTyWithBody {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(ast_passes::body_label)]
    pub body_span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::foreign_static_with_body)]
#[note(ast_passes::more_extern_note)]
pub struct ForeignStaticWithBody {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(ast_passes::body_label)]
    pub body_span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::foreign_fn_with_body)]
#[help]
#[note(ast_passes::more_extern_note)]
pub struct ForeignFnWithBody {
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion(code = ";", applicability = "maybe-incorrect")]
    pub body_span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::foreign_fn_with_qualifier)]
pub struct ForeignFnWithQualifier {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
    #[suggestion_verbose(code = "fn ", applicability = "maybe-incorrect")]
    pub replace_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::foreign_item_non_ascii)]
#[note]
pub struct ForeignItemNonAscii {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::forbidden_c_var_args)]
pub struct ForbiddenCVarArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::unnamed_assoc_const)]
pub struct UnnamedAssocConst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::nomangle_item_non_ascii, code = "E0754")]
pub struct NomangleItemNonAscii {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(ast_passes::mod_file_item_non_ascii, code = "E0754")]
#[help]
pub struct ModFileItemNonAscii {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}
