//! Errors emitted by ast_passes.

use rustc_ast::{visit::FnKind, ParamKindOrd};
use rustc_errors::{fluent, AddToDiagnostic, Applicability, Diagnostic, SubdiagnosticMessage};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

use crate::ast_validation::{DisallowTildeConstContext, ForbiddenLetReason};

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

#[derive(Diagnostic)]
#[diag(ast_passes::foreign_ty_with_generic_param)]
#[note(ast_passes::more_extern_note)]
pub struct ForeignTyWithGenericParam {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::foreign_ty_with_where_clause)]
#[note(ast_passes::more_extern_note)]
pub struct ForeignTyWithWhereClause {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
#[diag(ast_passes::foreign_fn_with_qualifier)]
pub struct ForeignFnWithQualifier {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
    #[suggestion_verbose(code = "fn ", applicability = "maybe-incorrect")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::foreign_item_non_ascii)]
#[note]
pub struct ForeignItemNonAscii {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes::extern_block_label)]
    pub extern_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::forbidden_c_var_args)]
pub struct ForbiddenCVarArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::unnamed_assoc_const)]
pub struct UnnamedAssocConst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::nomangle_item_non_ascii, code = "E0754")]
pub struct NomangleItemNonAscii {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::mod_file_item_non_ascii, code = "E0754")]
#[help]
pub struct ModFileItemNonAscii {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes::auto_trait_with_generic_param, code = "E0567")]
pub struct AutoTraitWithGenericParam {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[label(ast_passes::ident_label)]
    pub ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::auto_trait_with_super_trait_or_where_clause, code = "E0568")]
pub struct AutoTraitWithSuperTraitOrWhereClause {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[label(ast_passes::ident_label)]
    pub ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::auto_trait_with_assoc_item, code = "E0380")]
pub struct AutoTraitWithAssocItem {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub replace_span: Span,
    #[label(ast_passes::ident_label)]
    pub ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::generic_arg_after_constraint)]
pub struct GenericArgAfterConstraint {
    #[primary_span]
    pub arg_spans: Vec<Span>,
    #[label(ast_passes::constraints_label)]
    pub constraint_spans: Vec<Span>,
    #[label(ast_passes::last_arg_label)]
    pub last_arg_span: Span,
    #[label(ast_passes::first_constraint_label)]
    pub first_constraint_span: Span,
    #[suggestion_verbose(code = "{correct_order}", applicability = "machine-applicable")]
    pub replace_span: Span,
    pub args_len: usize,
    pub constraints_len: usize,
    pub correct_order: String,
}

#[derive(Diagnostic)]
#[diag(ast_passes::fn_ptr_ty_with_pat, code = "E0561")]
pub struct FnPtrTyWithPat {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::multiple_explicit_lifetime_bound, code = "E0226")]
pub struct MultipleExplicitLifetimeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::impl_trait_ty_in_path_param, code = "E0667")]
pub struct ImplTraitTyInPathParam {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::impl_trait_ty_nested, code = "E0666")]
pub struct ImplTraitTyNested {
    #[primary_span]
    #[label(ast_passes::nested_label)]
    pub nested_span: Span,
    #[label(ast_passes::outer_label)]
    pub outer_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::impl_trait_ty_without_trait_bound)]
pub struct ImplTraitTyWithoutTraitBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::generic_param_wrong_order)]
pub struct GenericParamWrongOrder {
    #[primary_span]
    pub spans: Vec<Span>,
    pub param_kind: ParamKindOrd,
    pub max_param_kind: ParamKindOrd,
    #[suggestion(code = "{correct_order}", applicability = "machine-applicable")]
    pub replace_span: Span,
    pub correct_order: String,
}

#[derive(Diagnostic)]
#[diag(ast_passes::obsolete_auto_trait_syntax)]
#[help]
pub struct ObsoleteAutoTraitSyntax {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::unsafe_negative_impl, code = "E0198")]
pub struct UnsafeNegativeImpl {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes::negative_label)]
    pub negative_span: Span,
    #[label(ast_passes::unsafe_label)]
    pub unsafe_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::unsafe_inherent_impl, code = "E0197")]
pub struct UnsafeInherentImpl {
    #[primary_span]
    #[label(ast_passes::ty_label)]
    pub span: Span,
    #[label(ast_passes::unsafe_label)]
    pub unsafe_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::negative_inherent_impl)]
pub struct NegativeInherentImpl {
    #[primary_span]
    #[label(ast_passes::ty_label)]
    pub span: Span,
    #[label(ast_passes::negative_label)]
    pub negative_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::default_inherent_impl)]
#[note]
pub struct DefaultInherentImpl {
    #[primary_span]
    #[label(ast_passes::ty_label)]
    pub span: Span,
    #[label(ast_passes::default_label)]
    pub default_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::const_inherent_impl)]
#[note]
pub struct ConstInherentImpl {
    #[primary_span]
    #[label(ast_passes::ty_label)]
    pub span: Span,
    #[label(ast_passes::const_label)]
    pub const_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::unsafe_extern_block)]
pub struct UnsafeExternBlock {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::unsafe_module)]
pub struct UnsafeModule {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::empty_union)]
pub struct EmptyUnion {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::ty_alias_with_where_clause)]
#[note]
pub struct TyAliasWithWhereClause {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::generic_param_with_default_not_trailing)]
pub struct GenericParamWithDefaultNotTrailing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::lifetime_nested_quantification, code = "E0316")]
pub struct LifetimeNestedQuantification {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::super_trait_with_maybe)]
#[note]
pub struct SuperTraitWithMaybe {
    #[primary_span]
    pub span: Span,
    pub path_str: String,
}

#[derive(Diagnostic)]
#[diag(ast_passes::trait_object_with_maybe)]
pub struct TraitObjectWithMaybe {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::forbidden_maybe_const)]
#[note]
pub struct ForbiddenMaybeConst<'a, 'b> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub(crate) reason: &'a DisallowTildeConstContext<'b>,
}

impl<'a> AddToDiagnostic for &'a DisallowTildeConstContext<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        match self {
            DisallowTildeConstContext::TraitObject => {
                diag.note(fluent::ast_passes::trait_object);
            }
            DisallowTildeConstContext::Fn(FnKind::Closure(..)) => {
                diag.note(fluent::ast_passes::closure);
            }
            DisallowTildeConstContext::Fn(FnKind::Fn(_, ident, ..)) => {
                diag.span_note(ident.span, fluent::ast_passes::fn_not_const);
            }
        }
    }
}

#[derive(Diagnostic)]
#[diag(ast_passes::maybe_const_with_maybe_trait)]
pub struct MaybeConstWithMaybeTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::const_async_fn)]
pub struct ConstAsyncFn {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label(ast_passes::const_label)]
    pub const_span: Span,
    #[label(ast_passes::async_label)]
    pub async_span: Span,
    #[label(ast_passes::fn_label)]
    pub fn_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::patterns_in_foreign_fns, code = "E0130")]
pub struct PatternsInForeignFns {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::patterns_in_fns_without_body, code = "E0642")]
pub struct PatternsInFnsWithoutBody {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes::equality_constraint)]
#[note]
pub struct EqualityConstraint {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub assoc_constraint_suggestion: Vec<EqualityConstraintToAssocConstraintSuggestion>,
}

pub struct EqualityConstraintToAssocConstraintSuggestion {
    pub assoc_ty: String,
    pub suggestion: Vec<(Span, String)>,
}

impl AddToDiagnostic for EqualityConstraintToAssocConstraintSuggestion {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        diag.set_arg("assoc_ty", self.assoc_ty);
        diag.multipart_suggestion(
            fluent::ast_passes::assoc_constraint_suggestion,
            self.suggestion,
            Applicability::MaybeIncorrect,
        );
    }
}
