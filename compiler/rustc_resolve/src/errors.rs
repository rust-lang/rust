use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{
    symbol::{Ident, Symbol},
    Span,
};

use crate::{late::PatternSource, Res};

#[derive(Diagnostic)]
#[diag(resolve_parent_module_reset_for_binding, code = "E0637")]
pub(crate) struct ParentModuleResetForBinding;

#[derive(Diagnostic)]
#[diag(resolve_ampersand_used_without_explicit_lifetime_name, code = "E0637")]
#[note]
pub(crate) struct AmpersandUsedWithoutExplicitLifetimeName(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag(resolve_underscore_lifetime_name_cannot_be_used_here, code = "E0637")]
#[note]
pub(crate) struct UnderscoreLifetimeNameCannotBeUsedHere(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag(resolve_crate_may_not_be_imported)]
pub(crate) struct CrateMayNotBeImported(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag(resolve_crate_root_imports_must_be_named_explicitly)]
pub(crate) struct CrateRootNamesMustBeNamedExplicitly(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag(resolve_crate_root_imports_must_be_named_explicitly)]
pub(crate) struct ResolutionError(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag(resolve_name_is_already_used_as_generic_parameter, code = "E0403")]
pub(crate) struct NameAlreadyUsedInParameterList {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(resolve_first_use_of_name)]
    pub(crate) first_use_span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_method_not_member_of_trait, code = "E0407")]
pub(crate) struct MethodNotMemberOfTrait {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) method: Ident,
    pub(crate) trait_: String,
    #[subdiagnostic]
    pub(crate) sub: Option<AssociatedFnWithSimilarNameExists>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    resolve_associated_fn_with_similar_name_exists,
    code = "{candidate}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedFnWithSimilarNameExists {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) candidate: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_type_not_member_of_trait, code = "E0437")]
pub(crate) struct TypeNotMemberOfTrait {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) type_: Ident,
    pub(crate) trait_: String,
    #[subdiagnostic]
    pub(crate) sub: Option<AssociatedTypeWithSimilarNameExists>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    resolve_associated_type_with_similar_name_exists,
    code = "{candidate}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedTypeWithSimilarNameExists {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) candidate: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_const_not_member_of_trait, code = "E0438")]
pub(crate) struct ConstNotMemberOfTrait {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) const_: Ident,
    pub(crate) trait_: String,
    #[subdiagnostic]
    pub(crate) sub: Option<AssociatedConstWithSimilarNameExists>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    resolve_associated_const_with_similar_name_exists,
    code = "{candidate}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedConstWithSimilarNameExists {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) candidate: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_variable_bound_with_different_mode, code = "E0409")]
pub(crate) struct VariableBoundWithDifferentMode {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(resolve_first_binding_span)]
    pub(crate) first_binding_span: Span,
    pub(crate) variable_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_ident_bound_more_than_once_in_parameter_list, code = "E0415")]
pub(crate) struct IdentifierBoundMoreThanOnceInParameterList {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) identifier: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_ident_bound_more_than_once_in_same_pattern, code = "E0416")]
pub(crate) struct IdentifierBoundMoreThanOnceInSamePattern {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) identifier: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_undeclared_label, code = "E0426")]
pub(crate) struct UndeclaredLabel {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[subdiagnostic]
    pub(crate) sub_reachable: Option<LabelWithSimilarNameReachable>,
    #[subdiagnostic]
    pub(crate) sub_reachable_suggestion: Option<TryUsingSimilarlyNamedLabel>,
    #[subdiagnostic]
    pub(crate) sub_unreachable: Option<UnreachableLabelWithSimilarNameExists>,
}

#[derive(Subdiagnostic)]
#[label(resolve_label_with_similar_name_reachable)]
pub(crate) struct LabelWithSimilarNameReachable(#[primary_span] pub(crate) Span);

#[derive(Subdiagnostic)]
#[suggestion(
    resolve_try_using_similarly_named_label,
    code = "{ident_name}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct TryUsingSimilarlyNamedLabel {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident_name: Symbol,
}

#[derive(Subdiagnostic)]
#[label(resolve_unreachable_label_with_similar_name_exists)]
pub(crate) struct UnreachableLabelWithSimilarNameExists {
    #[primary_span]
    pub(crate) ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_self_import_can_only_appear_once_in_the_list, code = "E0430")]
pub(crate) struct SelfImportCanOnlyAppearOnceInTheList {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_self_import_only_in_import_list_with_non_empty_prefix, code = "E0431")]
pub(crate) struct SelfImportOnlyInImportListWithNonEmptyPrefix {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_cannot_capture_dynamic_environment_in_fn_item, code = "E0434")]
#[help]
pub(crate) struct CannotCaptureDynamicEnvironmentInFnItem {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_attempt_to_use_non_constant_value_in_constant, code = "E0435")]
pub(crate) struct AttemptToUseNonConstantValueInConstant<'a> {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) with: Option<AttemptToUseNonConstantValueInConstantWithSuggestion<'a>>,
    #[subdiagnostic]
    pub(crate) with_label: Option<AttemptToUseNonConstantValueInConstantLabelWithSuggestion>,
    #[subdiagnostic]
    pub(crate) without: Option<AttemptToUseNonConstantValueInConstantWithoutSuggestion<'a>>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    resolve_attempt_to_use_non_constant_value_in_constant_with_suggestion,
    code = "{suggestion} {ident}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AttemptToUseNonConstantValueInConstantWithSuggestion<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
    pub(crate) suggestion: &'a str,
    pub(crate) current: &'a str,
}

#[derive(Subdiagnostic)]
#[label(resolve_attempt_to_use_non_constant_value_in_constant_label_with_suggestion)]
pub(crate) struct AttemptToUseNonConstantValueInConstantLabelWithSuggestion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[label(resolve_attempt_to_use_non_constant_value_in_constant_without_suggestion)]
pub(crate) struct AttemptToUseNonConstantValueInConstantWithoutSuggestion<'a> {
    #[primary_span]
    pub(crate) ident_span: Span,
    pub(crate) suggestion: &'a str,
}

#[derive(Diagnostic)]
#[diag(resolve_self_imports_only_allowed_within, code = "E0429")]
pub(crate) struct SelfImportsOnlyAllowedWithin {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) suggestion: Option<SelfImportsOnlyAllowedWithinSuggestion>,
    #[subdiagnostic]
    pub(crate) mpart_suggestion: Option<SelfImportsOnlyAllowedWithinMultipartSuggestion>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    resolve_self_imports_only_allowed_within_suggestion,
    code = "",
    applicability = "machine-applicable"
)]
pub(crate) struct SelfImportsOnlyAllowedWithinSuggestion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    resolve_self_imports_only_allowed_within_multipart_suggestion,
    applicability = "machine-applicable"
)]
pub(crate) struct SelfImportsOnlyAllowedWithinMultipartSuggestion {
    #[suggestion_part(code = "{{")]
    pub(crate) multipart_start: Span,
    #[suggestion_part(code = "}}")]
    pub(crate) multipart_end: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_binding_shadows_something_unacceptable, code = "E0530")]
pub(crate) struct BindingShadowsSomethingUnacceptable<'a> {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) shadowing_binding: PatternSource,
    pub(crate) shadowed_binding: Res,
    pub(crate) article: &'a str,
    #[subdiagnostic]
    pub(crate) sub_suggestion: Option<BindingShadowsSomethingUnacceptableSuggestion>,
    #[label(resolve_label_shadowed_binding)]
    pub(crate) shadowed_binding_span: Span,
    pub(crate) participle: &'a str,
    pub(crate) name: Symbol,
}

#[derive(Subdiagnostic)]
#[suggestion(
    resolve_binding_shadows_something_unacceptable_suggestion,
    code = "{name}(..)",
    applicability = "unspecified"
)]
pub(crate) struct BindingShadowsSomethingUnacceptableSuggestion {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_forward_declared_generic_param, code = "E0128")]
pub(crate) struct ForwardDeclaredGenericParam {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_param_in_ty_of_const_param, code = "E0770")]
pub(crate) struct ParamInTyOfConstParam {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[subdiagnostic]
    pub(crate) param_kind: Option<ParamKindInTyOfConstParam>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ParamKindInTyOfConstParam {
    #[note(resolve_type_param_in_ty_of_const_param)]
    Type,
    #[note(resolve_const_param_in_ty_of_const_param)]
    Const,
    #[note(resolve_lifetime_param_in_ty_of_const_param)]
    Lifetime,
}

#[derive(Diagnostic)]
#[diag(resolve_self_in_generic_param_default, code = "E0735")]
pub(crate) struct SelfInGenericParamDefault {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_param_in_non_trivial_anon_const)]
pub(crate) struct ParamInNonTrivialAnonConst {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[subdiagnostic]
    pub(crate) param_kind: ParamKindInNonTrivialAnonConst,
    #[subdiagnostic]
    pub(crate) help: Option<ParamInNonTrivialAnonConstHelp>,
}

#[derive(Subdiagnostic)]
#[help(resolve_param_in_non_trivial_anon_const_help)]
pub(crate) struct ParamInNonTrivialAnonConstHelp;

#[derive(Subdiagnostic)]
pub(crate) enum ParamKindInNonTrivialAnonConst {
    #[note(resolve_type_param_in_non_trivial_anon_const)]
    Type,
    #[help(resolve_const_param_in_non_trivial_anon_const)]
    Const { name: Symbol },
    #[note(resolve_lifetime_param_in_non_trivial_anon_const)]
    Lifetime,
}

#[derive(Diagnostic)]
#[diag(resolve_unreachable_label, code = "E0767")]
#[note]
pub(crate) struct UnreachableLabel {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[label(resolve_label_definition_span)]
    pub(crate) definition_span: Span,
    #[subdiagnostic]
    pub(crate) sub_suggestion: Option<UnreachableLabelSubSuggestion>,
    #[subdiagnostic]
    pub(crate) sub_suggestion_label: Option<UnreachableLabelSubLabel>,
    #[subdiagnostic]
    pub(crate) sub_unreachable_label: Option<UnreachableLabelSubLabelUnreachable>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    resolve_unreachable_label_suggestion_use_similarly_named,
    code = "{ident_name}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct UnreachableLabelSubSuggestion {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident_name: Symbol,
}

#[derive(Subdiagnostic)]
#[label(resolve_unreachable_label_similar_name_reachable)]
pub(crate) struct UnreachableLabelSubLabel {
    #[primary_span]
    pub(crate) ident_span: Span,
}

#[derive(Subdiagnostic)]
#[label(resolve_unreachable_label_similar_name_unreachable)]
pub(crate) struct UnreachableLabelSubLabelUnreachable {
    #[primary_span]
    pub(crate) ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_trait_impl_mismatch, code = "{code}")]
pub(crate) struct TraitImplMismatch {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    pub(crate) kind: String,
    #[label(resolve_label_trait_item)]
    pub(crate) trait_item_span: Span,
    pub(crate) trait_path: String,
    pub(crate) code: String,
}

#[derive(Diagnostic)]
#[diag(resolve_invalid_asm_sym)]
#[help]
pub(crate) struct InvalidAsmSym {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_trait_impl_duplicate, code = "E0201")]
pub(crate) struct TraitImplDuplicate {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(resolve_old_span_label)]
    pub(crate) old_span: Span,
    #[label(resolve_trait_item_span)]
    pub(crate) trait_item_span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_relative_2018)]
pub(crate) struct Relative2018 {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = "crate::{path_str}", applicability = "maybe-incorrect")]
    pub(crate) path_span: Span,
    pub(crate) path_str: String,
}

#[derive(Diagnostic)]
#[diag(resolve_ancestor_only, code = "E0742")]
pub(crate) struct AncestorOnly(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag(resolve_expected_found, code = "E0577")]
pub(crate) struct ExpectedFound {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) res: Res,
    pub(crate) path_str: String,
}

#[derive(Diagnostic)]
#[diag(resolve_indeterminate, code = "E0578")]
pub(crate) struct Indeterminate(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag(resolve_tool_module_imported)]
pub(crate) struct ToolModuleImported {
    #[primary_span]
    pub(crate) span: Span,
    #[note]
    pub(crate) import: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_module_only)]
pub(crate) struct ModuleOnly(#[primary_span] pub(crate) Span);

#[derive(Diagnostic, Default)]
#[diag(resolve_macro_expected_found)]
pub(crate) struct MacroExpectedFound<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) found: &'a str,
    pub(crate) expected: &'a str,
    pub(crate) macro_path: &'a str,
    #[subdiagnostic]
    pub(crate) remove_surrounding_derive: Option<RemoveSurroundingDerive>,
    #[subdiagnostic]
    pub(crate) add_as_non_derive: Option<AddAsNonDerive<'a>>,
}

#[derive(Subdiagnostic)]
#[help(resolve_remove_surrounding_derive)]
pub(crate) struct RemoveSurroundingDerive {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[help(resolve_add_as_non_derive)]
pub(crate) struct AddAsNonDerive<'a> {
    pub(crate) macro_path: &'a str,
}

#[derive(Diagnostic)]
#[diag(resolve_proc_macro_same_crate)]
pub(crate) struct ProcMacroSameCrate {
    #[primary_span]
    pub(crate) span: Span,
    #[help]
    pub(crate) is_test: bool,
}

#[derive(Diagnostic)]
#[diag(resolve_imported_crate)]
pub(crate) struct CrateImported {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_macro_use_extern_crate_self)]
pub(crate) struct MacroUseExternCrateSelf {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_accessible_unsure)]
#[note]
pub(crate) struct CfgAccessibleUnsure {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(resolve_param_in_enum_discriminant)]
pub(crate) struct ParamInEnumDiscriminant {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[subdiagnostic]
    pub(crate) param_kind: ParamKindInEnumDiscriminant,
}

#[derive(Subdiagnostic)]
pub(crate) enum ParamKindInEnumDiscriminant {
    #[note(resolve_type_param_in_enum_discriminant)]
    Type,
    #[note(resolve_const_param_in_enum_discriminant)]
    Const,
    #[note(resolve_lifetime_param_in_enum_discriminant)]
    Lifetime,
}
