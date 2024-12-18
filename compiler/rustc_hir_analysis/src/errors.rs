//! Errors emitted by `rustc_hir_analysis`.

use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level, MultiSpan,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::{Ident, Span, Symbol};

use crate::fluent_generated as fluent;
mod pattern_types;
pub(crate) use pattern_types::*;
pub(crate) mod wrong_number_of_generic_args;

mod precise_captures;
pub(crate) use precise_captures::*;

#[derive(Diagnostic)]
#[diag(hir_analysis_ambiguous_assoc_item)]
pub(crate) struct AmbiguousAssocItem<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub assoc_kind: &'static str,
    pub assoc_name: Ident,
    pub qself: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_assoc_kind_mismatch)]
pub(crate) struct AssocKindMismatch {
    #[primary_span]
    #[label]
    pub span: Span,
    pub expected: &'static str,
    pub got: &'static str,
    #[label(hir_analysis_expected_because_label)]
    pub expected_because_label: Option<Span>,
    pub assoc_kind: &'static str,
    #[note]
    pub def_span: Span,
    #[label(hir_analysis_bound_on_assoc_const_label)]
    pub bound_on_assoc_const_label: Option<Span>,
    #[subdiagnostic]
    pub wrap_in_braces_sugg: Option<AssocKindMismatchWrapInBracesSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_analysis_assoc_kind_mismatch_wrap_in_braces_sugg,
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssocKindMismatchWrapInBracesSugg {
    #[suggestion_part(code = "{{ ")]
    pub lo: Span,
    #[suggestion_part(code = " }}")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_assoc_item_is_private, code = E0624)]
pub(crate) struct AssocItemIsPrivate {
    #[primary_span]
    #[label]
    pub span: Span,
    pub kind: &'static str,
    pub name: Ident,
    #[label(hir_analysis_defined_here_label)]
    pub defined_here_label: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_assoc_item_not_found, code = E0220)]
pub(crate) struct AssocItemNotFound<'a> {
    #[primary_span]
    pub span: Span,
    pub assoc_name: Ident,
    pub assoc_kind: &'static str,
    pub qself: &'a str,
    #[subdiagnostic]
    pub label: Option<AssocItemNotFoundLabel<'a>>,
    #[subdiagnostic]
    pub sugg: Option<AssocItemNotFoundSugg<'a>>,
}

#[derive(Subdiagnostic)]
pub(crate) enum AssocItemNotFoundLabel<'a> {
    #[label(hir_analysis_assoc_item_not_found_label)]
    NotFound {
        #[primary_span]
        span: Span,
    },
    #[label(hir_analysis_assoc_item_not_found_found_in_other_trait_label)]
    FoundInOtherTrait {
        #[primary_span]
        span: Span,
        assoc_kind: &'static str,
        trait_name: &'a str,
        suggested_name: Symbol,
        identically_named: bool,
    },
}

#[derive(Subdiagnostic)]

pub(crate) enum AssocItemNotFoundSugg<'a> {
    #[suggestion(
        hir_analysis_assoc_item_not_found_similar_sugg,
        code = "{suggested_name}",
        applicability = "maybe-incorrect"
    )]
    Similar {
        #[primary_span]
        span: Span,
        assoc_kind: &'static str,
        suggested_name: Symbol,
    },
    #[suggestion(
        hir_analysis_assoc_item_not_found_similar_in_other_trait_sugg,
        code = "{suggested_name}",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    SimilarInOtherTrait {
        #[primary_span]
        span: Span,
        assoc_kind: &'static str,
        suggested_name: Symbol,
    },
    #[multipart_suggestion(
        hir_analysis_assoc_item_not_found_similar_in_other_trait_qpath_sugg,
        style = "verbose"
    )]
    SimilarInOtherTraitQPath {
        #[suggestion_part(code = "<")]
        lo: Span,
        #[suggestion_part(code = " as {trait_ref}>")]
        mi: Span,
        #[suggestion_part(code = "{suggested_name}")]
        hi: Option<Span>,
        trait_ref: String,
        suggested_name: Symbol,
        identically_named: bool,
        #[applicability]
        applicability: Applicability,
    },
    #[suggestion(
        hir_analysis_assoc_item_not_found_other_sugg,
        code = "{suggested_name}",
        applicability = "maybe-incorrect"
    )]
    Other {
        #[primary_span]
        span: Span,
        qself: &'a str,
        assoc_kind: &'static str,
        suggested_name: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag(hir_analysis_unrecognized_atomic_operation, code = E0092)]
pub(crate) struct UnrecognizedAtomicOperation<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub op: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_wrong_number_of_generic_arguments_to_intrinsic, code = E0094)]
pub(crate) struct WrongNumberOfGenericArgumentsToIntrinsic<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub found: usize,
    pub expected: usize,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_unrecognized_intrinsic_function, code = E0093)]
#[help]
pub(crate) struct UnrecognizedIntrinsicFunction {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_lifetimes_or_bounds_mismatch_on_trait, code = E0195)]
pub(crate) struct LifetimesOrBoundsMismatchOnTrait {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_analysis_generics_label)]
    pub generics_span: Option<Span>,
    #[label(hir_analysis_where_label)]
    pub where_span: Option<Span>,
    #[label(hir_analysis_bounds_label)]
    pub bounds_span: Vec<Span>,
    pub item_kind: &'static str,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_drop_impl_on_wrong_item, code = E0120)]
pub(crate) struct DropImplOnWrongItem {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum FieldAlreadyDeclared {
    #[diag(hir_analysis_field_already_declared, code = E0124)]
    NotNested {
        field_name: Symbol,
        #[primary_span]
        #[label]
        span: Span,
        #[label(hir_analysis_previous_decl_label)]
        prev_span: Span,
    },
    #[diag(hir_analysis_field_already_declared_current_nested)]
    CurrentNested {
        field_name: Symbol,
        #[primary_span]
        #[label]
        span: Span,
        #[note(hir_analysis_nested_field_decl_note)]
        nested_field_span: Span,
        #[subdiagnostic]
        help: FieldAlreadyDeclaredNestedHelp,
        #[label(hir_analysis_previous_decl_label)]
        prev_span: Span,
    },
    #[diag(hir_analysis_field_already_declared_previous_nested)]
    PreviousNested {
        field_name: Symbol,
        #[primary_span]
        #[label]
        span: Span,
        #[label(hir_analysis_previous_decl_label)]
        prev_span: Span,
        #[note(hir_analysis_previous_nested_field_decl_note)]
        prev_nested_field_span: Span,
        #[subdiagnostic]
        prev_help: FieldAlreadyDeclaredNestedHelp,
    },
    #[diag(hir_analysis_field_already_declared_both_nested)]
    BothNested {
        field_name: Symbol,
        #[primary_span]
        #[label]
        span: Span,
        #[note(hir_analysis_nested_field_decl_note)]
        nested_field_span: Span,
        #[subdiagnostic]
        help: FieldAlreadyDeclaredNestedHelp,
        #[label(hir_analysis_previous_decl_label)]
        prev_span: Span,
        #[note(hir_analysis_previous_nested_field_decl_note)]
        prev_nested_field_span: Span,
        #[subdiagnostic]
        prev_help: FieldAlreadyDeclaredNestedHelp,
    },
}

#[derive(Subdiagnostic)]
#[help(hir_analysis_field_already_declared_nested_help)]
pub(crate) struct FieldAlreadyDeclaredNestedHelp {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_copy_impl_on_type_with_dtor, code = E0184)]
pub(crate) struct CopyImplOnTypeWithDtor {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_multiple_relaxed_default_bounds, code = E0203)]
pub(crate) struct MultipleRelaxedDefaultBounds {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_copy_impl_on_non_adt, code = E0206)]
pub(crate) struct CopyImplOnNonAdt {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_const_param_ty_impl_on_unsized)]
pub(crate) struct ConstParamTyImplOnUnsized {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_const_param_ty_impl_on_non_adt)]
pub(crate) struct ConstParamTyImplOnNonAdt {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_trait_object_declared_with_no_traits, code = E0224)]
pub(crate) struct TraitObjectDeclaredWithNoTraits {
    #[primary_span]
    pub span: Span,
    #[label(hir_analysis_alias_span)]
    pub trait_alias_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_ambiguous_lifetime_bound, code = E0227)]
pub(crate) struct AmbiguousLifetimeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_assoc_item_constraints_not_allowed_here, code = E0229)]
pub(crate) struct AssocItemConstraintsNotAllowedHere {
    #[primary_span]
    #[label]
    pub span: Span,

    #[subdiagnostic]
    pub fn_trait_expansion: Option<ParenthesizedFnTraitExpansion>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_param_in_ty_of_assoc_const_binding)]
pub(crate) struct ParamInTyOfAssocConstBinding<'tcx> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub assoc_const: Ident,
    pub param_name: Symbol,
    pub param_def_kind: &'static str,
    pub param_category: &'static str,
    #[label(hir_analysis_param_defined_here_label)]
    pub param_defined_here_label: Option<Span>,
    #[subdiagnostic]
    pub ty_note: Option<TyOfAssocConstBindingNote<'tcx>>,
}

#[derive(Subdiagnostic, Clone, Copy)]
#[note(hir_analysis_ty_of_assoc_const_binding_note)]
pub(crate) struct TyOfAssocConstBindingNote<'tcx> {
    pub assoc_const: Ident,
    pub ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_escaping_bound_var_in_ty_of_assoc_const_binding)]
pub(crate) struct EscapingBoundVarInTyOfAssocConstBinding<'tcx> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub assoc_const: Ident,
    pub var_name: Symbol,
    pub var_def_kind: &'static str,
    #[label(hir_analysis_var_defined_here_label)]
    pub var_defined_here_label: Span,
    #[subdiagnostic]
    pub ty_note: Option<TyOfAssocConstBindingNote<'tcx>>,
}

#[derive(Subdiagnostic)]
#[help(hir_analysis_parenthesized_fn_trait_expansion)]
pub(crate) struct ParenthesizedFnTraitExpansion {
    #[primary_span]
    pub span: Span,

    pub expanded_type: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_typeof_reserved_keyword_used, code = E0516)]
pub(crate) struct TypeofReservedKeywordUsed<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion(style = "verbose", code = "{ty}")]
    pub opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_value_of_associated_struct_already_specified, code = E0719)]
pub(crate) struct ValueOfAssociatedStructAlreadySpecified {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_analysis_previous_bound_label)]
    pub prev_span: Span,
    pub item_name: Ident,
    pub def_path: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_unconstrained_opaque_type)]
#[note]
pub(crate) struct UnconstrainedOpaqueType {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub what: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_tait_forward_compat)]
#[note]
pub(crate) struct TaitForwardCompat {
    #[primary_span]
    pub span: Span,
    #[note]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_tait_forward_compat2)]
#[note]
pub(crate) struct TaitForwardCompat2 {
    #[primary_span]
    pub span: Span,
    #[note(hir_analysis_opaque)]
    pub opaque_type_span: Span,
    pub opaque_type: String,
}

pub(crate) struct MissingTypeParams {
    pub span: Span,
    pub def_span: Span,
    pub span_snippet: Option<String>,
    pub missing_type_params: Vec<Symbol>,
    pub empty_generic_args: bool,
}

// Manual implementation of `Diagnostic` to be able to call `span_to_snippet`.
impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for MissingTypeParams {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut err = Diag::new(dcx, level, fluent::hir_analysis_missing_type_params);
        err.span(self.span);
        err.code(E0393);
        err.arg("parameterCount", self.missing_type_params.len());
        err.arg(
            "parameters",
            self.missing_type_params
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", "),
        );

        err.span_label(self.def_span, fluent::hir_analysis_label);

        let mut suggested = false;
        // Don't suggest setting the type params if there are some already: the order is
        // tricky to get right and the user will already know what the syntax is.
        if let Some(snippet) = self.span_snippet
            && self.empty_generic_args
        {
            if snippet.ends_with('>') {
                // The user wrote `Trait<'a, T>` or similar. To provide an accurate suggestion
                // we would have to preserve the right order. For now, as clearly the user is
                // aware of the syntax, we do nothing.
            } else {
                // The user wrote `Iterator`, so we don't have a type we can suggest, but at
                // least we can clue them to the correct syntax `Iterator<Type>`.
                err.span_suggestion_verbose(
                    self.span.shrink_to_hi(),
                    fluent::hir_analysis_suggestion,
                    format!(
                        "<{}>",
                        self.missing_type_params
                            .iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    Applicability::HasPlaceholders,
                );
                suggested = true;
            }
        }
        if !suggested {
            err.span_label(self.span, fluent::hir_analysis_no_suggestion_label);
        }

        err.note(fluent::hir_analysis_note);
        err
    }
}

#[derive(Diagnostic)]
#[diag(hir_analysis_manual_implementation, code = E0183)]
#[help]
pub(crate) struct ManualImplementation {
    #[primary_span]
    #[label]
    pub span: Span,
    pub trait_name: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_generic_args_on_overridden_impl)]
pub(crate) struct GenericArgsOnOverriddenImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_const_impl_for_non_const_trait)]
pub(crate) struct ConstImplForNonConstTrait {
    #[primary_span]
    #[label]
    pub trait_ref_span: Span,
    pub trait_name: String,
    #[suggestion(
        applicability = "machine-applicable",
        code = "#[const_trait] ",
        style = "verbose"
    )]
    pub local_trait_span: Option<Span>,
    pub suggestion_pre: &'static str,
    #[note]
    pub marking: (),
    #[note(hir_analysis_adding)]
    pub adding: (),
}

#[derive(Diagnostic)]
#[diag(hir_analysis_const_bound_for_non_const_trait)]
pub(crate) struct ConstBoundForNonConstTrait {
    #[primary_span]
    #[label]
    pub span: Span,
    pub modifier: &'static str,
    #[note]
    pub def_span: Option<Span>,
    pub suggestion_pre: &'static str,
    #[suggestion(
        applicability = "machine-applicable",
        code = "#[const_trait] ",
        style = "verbose"
    )]
    pub suggestion: Option<Span>,
    pub trait_name: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_self_in_impl_self)]
pub(crate) struct SelfInImplSelf {
    #[primary_span]
    pub span: MultiSpan,
    #[note]
    pub note: (),
}

#[derive(Diagnostic)]
#[diag(hir_analysis_linkage_type, code = E0791)]
pub(crate) struct LinkageType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[help]
#[diag(hir_analysis_auto_deref_reached_recursion_limit, code = E0055)]
pub(crate) struct AutoDerefReachedRecursionLimit<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub ty: Ty<'a>,
    pub suggested_limit: rustc_session::Limit,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_where_clause_on_main, code = E0646)]
pub(crate) struct WhereClauseOnMain {
    #[primary_span]
    pub span: Span,
    #[label]
    pub generics_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_track_caller_on_main)]
pub(crate) struct TrackCallerOnMain {
    #[primary_span]
    #[suggestion(applicability = "maybe-incorrect", code = "")]
    pub span: Span,
    #[label(hir_analysis_track_caller_on_main)]
    pub annotated: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_target_feature_on_main)]
pub(crate) struct TargetFeatureOnMain {
    #[primary_span]
    #[label(hir_analysis_target_feature_on_main)]
    pub main: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_start_not_track_caller)]
pub(crate) struct StartTrackCaller {
    #[primary_span]
    pub span: Span,
    #[label]
    pub start: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_start_not_target_feature)]
pub(crate) struct StartTargetFeature {
    #[primary_span]
    pub span: Span,
    #[label]
    pub start: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_start_not_async, code = E0752)]
pub(crate) struct StartAsync {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_start_function_where, code = E0647)]
pub(crate) struct StartFunctionWhere {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_start_function_parameters, code = E0132)]
pub(crate) struct StartFunctionParameters {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_main_function_return_type_generic, code = E0131)]
pub(crate) struct MainFunctionReturnTypeGeneric {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_main_function_async, code = E0752)]
pub(crate) struct MainFunctionAsync {
    #[primary_span]
    pub span: Span,
    #[label]
    pub asyncness: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_main_function_generic_parameters, code = E0131)]
pub(crate) struct MainFunctionGenericParameters {
    #[primary_span]
    pub span: Span,
    #[label]
    pub label_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_variadic_function_compatible_convention, code = E0045)]
pub(crate) struct VariadicFunctionCompatibleConvention {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum CannotCaptureLateBound {
    #[diag(hir_analysis_cannot_capture_late_bound_ty)]
    Type {
        #[primary_span]
        use_span: Span,
        #[label]
        def_span: Span,
        what: &'static str,
    },
    #[diag(hir_analysis_cannot_capture_late_bound_const)]
    Const {
        #[primary_span]
        use_span: Span,
        #[label]
        def_span: Span,
        what: &'static str,
    },
    #[diag(hir_analysis_cannot_capture_late_bound_lifetime)]
    Lifetime {
        #[primary_span]
        use_span: Span,
        #[label]
        def_span: Span,
        what: &'static str,
    },
}

#[derive(Diagnostic)]
#[diag(hir_analysis_variances_of)]
pub(crate) struct VariancesOf {
    #[primary_span]
    pub span: Span,
    pub variances: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_type_of)]
pub(crate) struct TypeOf<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_invalid_union_field, code = E0740)]
pub(crate) struct InvalidUnionField {
    #[primary_span]
    pub field_span: Span,
    #[subdiagnostic]
    pub sugg: InvalidUnionFieldSuggestion,
    #[note]
    pub note: (),
}

#[derive(Diagnostic)]
#[diag(hir_analysis_invalid_unsafe_field, code = E0740)]
pub(crate) struct InvalidUnsafeField {
    #[primary_span]
    pub field_span: Span,
    #[subdiagnostic]
    pub sugg: InvalidUnsafeFieldSuggestion,
    #[note]
    pub note: (),
}

#[derive(Diagnostic)]
#[diag(hir_analysis_return_type_notation_on_non_rpitit)]
pub(crate) struct ReturnTypeNotationOnNonRpitit<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    #[label]
    pub fn_span: Option<Span>,
    #[note]
    pub note: (),
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(hir_analysis_invalid_union_field_sugg, applicability = "machine-applicable")]
pub(crate) struct InvalidUnionFieldSuggestion {
    #[suggestion_part(code = "std::mem::ManuallyDrop<")]
    pub lo: Span,
    #[suggestion_part(code = ">")]
    pub hi: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_analysis_invalid_unsafe_field_sugg,
    applicability = "machine-applicable"
)]
pub(crate) struct InvalidUnsafeFieldSuggestion {
    #[suggestion_part(code = "std::mem::ManuallyDrop<")]
    pub lo: Span,
    #[suggestion_part(code = ">")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_return_type_notation_equality_bound)]
pub(crate) struct ReturnTypeNotationEqualityBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_placeholder_not_allowed_item_signatures, code = E0121)]
pub(crate) struct PlaceholderNotAllowedItemSignatures {
    #[primary_span]
    #[label]
    pub spans: Vec<Span>,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_associated_type_trait_uninferred_generic_params, code = E0212)]
pub(crate) struct AssociatedItemTraitUninferredGenericParams {
    #[primary_span]
    pub span: Span,
    #[suggestion(style = "verbose", applicability = "maybe-incorrect", code = "{bound}")]
    pub inferred_sugg: Option<Span>,
    pub bound: String,
    #[subdiagnostic]
    pub mpart_sugg: Option<AssociatedItemTraitUninferredGenericParamsMultipartSuggestion>,
    pub what: &'static str,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_analysis_associated_type_trait_uninferred_generic_params_multipart_suggestion,
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedItemTraitUninferredGenericParamsMultipartSuggestion {
    #[suggestion_part(code = "{first}")]
    pub fspan: Span,
    pub first: String,
    #[suggestion_part(code = "{second}")]
    pub sspan: Span,
    pub second: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_enum_discriminant_overflowed, code = E0370)]
#[note]
pub(crate) struct EnumDiscriminantOverflowed {
    #[primary_span]
    #[label]
    pub span: Span,
    pub discr: String,
    pub item_name: Symbol,
    pub wrapped_discr: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_paren_sugar_attribute)]
#[help]
pub(crate) struct ParenSugarAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_must_implement_one_of_attribute)]
pub(crate) struct MustImplementOneOfAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_must_be_name_of_associated_function)]
pub(crate) struct MustBeNameOfAssociatedFunction {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_function_not_have_default_implementation)]
pub(crate) struct FunctionNotHaveDefaultImplementation {
    #[primary_span]
    pub span: Span,
    #[note]
    pub note_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_must_implement_not_function)]
pub(crate) struct MustImplementNotFunction {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub span_note: MustImplementNotFunctionSpanNote,
    #[subdiagnostic]
    pub note: MustImplementNotFunctionNote,
}

#[derive(Subdiagnostic)]
#[note(hir_analysis_must_implement_not_function_span_note)]
pub(crate) struct MustImplementNotFunctionSpanNote {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[note(hir_analysis_must_implement_not_function_note)]
pub(crate) struct MustImplementNotFunctionNote {}

#[derive(Diagnostic)]
#[diag(hir_analysis_function_not_found_in_trait)]
pub(crate) struct FunctionNotFoundInTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_functions_names_duplicated)]
#[note]
pub(crate) struct FunctionNamesDuplicated {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_simd_ffi_highly_experimental)]
#[help]
pub(crate) struct SIMDFFIHighlyExperimental {
    #[primary_span]
    pub span: Span,
    pub snip: String,
}

#[derive(Diagnostic)]
pub(crate) enum ImplNotMarkedDefault {
    #[diag(hir_analysis_impl_not_marked_default, code = E0520)]
    #[note]
    Ok {
        #[primary_span]
        #[label]
        span: Span,
        #[label(hir_analysis_ok_label)]
        ok_label: Span,
        ident: Symbol,
    },
    #[diag(hir_analysis_impl_not_marked_default_err, code = E0520)]
    #[note]
    Err {
        #[primary_span]
        span: Span,
        cname: Symbol,
        ident: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag(hir_analysis_missing_trait_item, code = E0046)]
pub(crate) struct MissingTraitItem {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub missing_trait_item_label: Vec<MissingTraitItemLabel>,
    #[subdiagnostic]
    pub missing_trait_item: Vec<MissingTraitItemSuggestion>,
    #[subdiagnostic]
    pub missing_trait_item_none: Vec<MissingTraitItemSuggestionNone>,
    pub missing_items_msg: String,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_missing_trait_item_label)]
pub(crate) struct MissingTraitItemLabel {
    #[primary_span]
    pub span: Span,
    pub item: Symbol,
}

#[derive(Subdiagnostic)]
#[suggestion(
    hir_analysis_missing_trait_item_suggestion,
    style = "tool-only",
    applicability = "has-placeholders",
    code = "{code}"
)]
pub(crate) struct MissingTraitItemSuggestion {
    #[primary_span]
    pub span: Span,
    pub code: String,
    pub snippet: String,
}

#[derive(Subdiagnostic)]
#[suggestion(
    hir_analysis_missing_trait_item_suggestion,
    style = "hidden",
    applicability = "has-placeholders",
    code = "{code}"
)]
pub(crate) struct MissingTraitItemSuggestionNone {
    #[primary_span]
    pub span: Span,
    pub code: String,
    pub snippet: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_missing_one_of_trait_item, code = E0046)]
pub(crate) struct MissingOneOfTraitItem {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note]
    pub note: Option<Span>,
    pub missing_items_msg: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_missing_trait_item_unstable, code = E0046)]
#[note]
pub(crate) struct MissingTraitItemUnstable {
    #[primary_span]
    pub span: Span,
    #[note(hir_analysis_some_note)]
    pub some_note: bool,
    #[note(hir_analysis_none_note)]
    pub none_note: bool,
    pub missing_item_name: Symbol,
    pub feature: Symbol,
    pub reason: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_transparent_enum_variant, code = E0731)]
pub(crate) struct TransparentEnumVariant {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_analysis_multi_label)]
    pub spans: Vec<Span>,
    #[label(hir_analysis_many_label)]
    pub many: Option<Span>,
    pub number: usize,
    pub path: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_transparent_non_zero_sized_enum, code = E0690)]
pub(crate) struct TransparentNonZeroSizedEnum<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_analysis_labels)]
    pub spans: Vec<Span>,
    pub field_count: usize,
    pub desc: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_transparent_non_zero_sized, code = E0690)]
pub(crate) struct TransparentNonZeroSized<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_analysis_labels)]
    pub spans: Vec<Span>,
    pub field_count: usize,
    pub desc: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_too_large_static)]
pub(crate) struct TooLargeStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_specialization_trait)]
#[help]
pub(crate) struct SpecializationTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_closure_implicit_hrtb)]
pub(crate) struct ClosureImplicitHrtb {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label]
    pub for_sp: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_empty_specialization)]
pub(crate) struct EmptySpecialization {
    #[primary_span]
    pub span: Span,
    #[note]
    pub base_impl_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_static_specialize)]
pub(crate) struct StaticSpecialize {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum DropImplPolarity {
    #[diag(hir_analysis_drop_impl_negative)]
    Negative {
        #[primary_span]
        span: Span,
    },
    #[diag(hir_analysis_drop_impl_reservation)]
    Reservation {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
pub(crate) enum ReturnTypeNotationIllegalParam {
    #[diag(hir_analysis_return_type_notation_illegal_param_type)]
    Type {
        #[primary_span]
        span: Span,
        #[label]
        param_span: Span,
    },
    #[diag(hir_analysis_return_type_notation_illegal_param_const)]
    Const {
        #[primary_span]
        span: Span,
        #[label]
        param_span: Span,
    },
}

#[derive(Diagnostic)]
pub(crate) enum LateBoundInApit {
    #[diag(hir_analysis_late_bound_type_in_apit)]
    Type {
        #[primary_span]
        span: Span,
        #[label]
        param_span: Span,
    },
    #[diag(hir_analysis_late_bound_const_in_apit)]
    Const {
        #[primary_span]
        span: Span,
        #[label]
        param_span: Span,
    },
    #[diag(hir_analysis_late_bound_lifetime_in_apit)]
    Lifetime {
        #[primary_span]
        span: Span,
        #[label]
        param_span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(hir_analysis_unused_associated_type_bounds)]
#[note]
pub(crate) struct UnusedAssociatedTypeBounds {
    #[suggestion(code = "")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(hir_analysis_rpitit_refined)]
#[note]
#[note(hir_analysis_feedback_note)]
pub(crate) struct ReturnPositionImplTraitInTraitRefined<'tcx> {
    #[suggestion(applicability = "maybe-incorrect", code = "{pre}{return_ty}{post}")]
    pub impl_return_span: Span,
    #[label]
    pub trait_return_span: Option<Span>,
    #[label(hir_analysis_unmatched_bound_label)]
    pub unmatched_bound: Option<Span>,

    pub pre: &'static str,
    pub post: &'static str,
    pub return_ty: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(hir_analysis_rpitit_refined_lifetimes)]
#[note]
#[note(hir_analysis_feedback_note)]
pub(crate) struct ReturnPositionImplTraitInTraitRefinedLifetimes {
    #[suggestion(applicability = "maybe-incorrect", code = "{suggestion}")]
    pub suggestion_span: Span,
    pub suggestion: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_ty_outside, code = E0390)]
#[help]
pub(crate) struct InherentTyOutside {
    #[primary_span]
    #[help(hir_analysis_span_help)]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0378)]
pub(crate) struct DispatchFromDynCoercion<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
    #[note(hir_analysis_coercion_between_struct_same_note)]
    pub note: bool,
    pub source_path: String,
    pub target_path: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_dispatch_from_dyn_repr, code = E0378)]
pub(crate) struct DispatchFromDynRepr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_ty_outside_relevant, code = E0390)]
#[help]
pub(crate) struct InherentTyOutsideRelevant {
    #[primary_span]
    pub span: Span,
    #[help(hir_analysis_span_help)]
    pub help_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_ty_outside_new, code = E0116)]
#[note]
pub(crate) struct InherentTyOutsideNew {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_ty_outside_primitive, code = E0390)]
#[help]
pub(crate) struct InherentTyOutsidePrimitive {
    #[primary_span]
    pub span: Span,
    #[help(hir_analysis_span_help)]
    pub help_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_primitive_ty, code = E0390)]
#[help]
pub(crate) struct InherentPrimitiveTy<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub note: Option<InherentPrimitiveTyNote<'a>>,
}

#[derive(Subdiagnostic)]
#[note(hir_analysis_inherent_primitive_ty_note)]
pub(crate) struct InherentPrimitiveTyNote<'a> {
    pub subty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_dyn, code = E0785)]
#[note]
pub(crate) struct InherentDyn {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_nominal, code = E0118)]
#[note]
pub(crate) struct InherentNominal {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_dispatch_from_dyn_zst, code = E0378)]
#[note]
pub(crate) struct DispatchFromDynZST<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0378)]
pub(crate) struct DispatchFromDynSingle<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
    #[note(hir_analysis_coercion_between_struct_single_note)]
    pub note: bool,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_dispatch_from_dyn_multi, code = E0378)]
#[note]
pub(crate) struct DispatchFromDynMulti {
    #[primary_span]
    pub span: Span,
    #[note(hir_analysis_coercions_note)]
    pub coercions_note: bool,
    pub number: usize,
    pub coercions: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0376)]
pub(crate) struct DispatchFromDynStruct<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0377)]
pub(crate) struct DispatchFromDynSame<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
    #[note(hir_analysis_coercion_between_struct_same_note)]
    pub note: bool,
    pub source_path: String,
    pub target_path: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0374)]
pub(crate) struct CoerceUnsizedOneField<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
    #[note(hir_analysis_coercion_between_struct_single_note)]
    pub note: bool,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_multi, code = E0375)]
#[note]
pub(crate) struct CoerceUnsizedMulti {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note(hir_analysis_coercions_note)]
    pub coercions_note: bool,
    pub number: usize,
    pub coercions: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0378)]
pub(crate) struct CoerceUnsizedMay<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_trait_cannot_impl_for_ty, code = E0204)]
pub(crate) struct TraitCannotImplForTy {
    #[primary_span]
    pub span: Span,
    pub trait_name: String,
    #[label]
    pub label_spans: Vec<Span>,
    #[subdiagnostic]
    pub notes: Vec<ImplForTyRequires>,
}

#[derive(Subdiagnostic)]
#[note(hir_analysis_requires_note)]
pub(crate) struct ImplForTyRequires {
    #[primary_span]
    pub span: MultiSpan,
    pub error_predicate: String,
    pub trait_name: String,
    pub ty: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_traits_with_defualt_impl, code = E0321)]
#[note]
pub(crate) struct TraitsWithDefaultImpl<'a> {
    #[primary_span]
    pub span: Span,
    pub traits: String,
    pub problematic_kind: &'a str,
    pub self_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cross_crate_traits, code = E0321)]
pub(crate) struct CrossCrateTraits<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub traits: String,
    pub self_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cross_crate_traits_defined, code = E0321)]
pub(crate) struct CrossCrateTraitsDefined {
    #[primary_span]
    #[label]
    pub span: Span,
    pub traits: String,
}

// FIXME(fmease): Deduplicate:

#[derive(Diagnostic)]
#[diag(hir_analysis_ty_param_first_local, code = E0210)]
#[note]
pub(crate) struct TyParamFirstLocal<'tcx> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note(hir_analysis_case_note)]
    pub note: (),
    pub param: Symbol,
    pub local_type: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(hir_analysis_ty_param_first_local, code = E0210)]
#[note]
pub(crate) struct TyParamFirstLocalLint<'tcx> {
    #[label]
    pub span: Span,
    #[note(hir_analysis_case_note)]
    pub note: (),
    pub param: Symbol,
    pub local_type: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_ty_param_some, code = E0210)]
#[note]
pub(crate) struct TyParamSome {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note(hir_analysis_only_note)]
    pub note: (),
    pub param: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(hir_analysis_ty_param_some, code = E0210)]
#[note]
pub(crate) struct TyParamSomeLint {
    #[label]
    pub span: Span,
    #[note(hir_analysis_only_note)]
    pub note: (),
    pub param: Symbol,
}

#[derive(Diagnostic)]
pub(crate) enum OnlyCurrentTraits {
    #[diag(hir_analysis_only_current_traits_outside, code = E0117)]
    Outside {
        #[primary_span]
        span: Span,
        #[note(hir_analysis_only_current_traits_note_uncovered)]
        #[note(hir_analysis_only_current_traits_note_more_info)]
        #[note(hir_analysis_only_current_traits_note)]
        note: (),
    },
    #[diag(hir_analysis_only_current_traits_primitive, code = E0117)]
    Primitive {
        #[primary_span]
        span: Span,
        #[note(hir_analysis_only_current_traits_note_uncovered)]
        #[note(hir_analysis_only_current_traits_note_more_info)]
        #[note(hir_analysis_only_current_traits_note)]
        note: (),
    },
    #[diag(hir_analysis_only_current_traits_arbitrary, code = E0117)]
    Arbitrary {
        #[primary_span]
        span: Span,
        #[note(hir_analysis_only_current_traits_note_uncovered)]
        #[note(hir_analysis_only_current_traits_note_more_info)]
        #[note(hir_analysis_only_current_traits_note)]
        note: (),
    },
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_opaque)]
pub(crate) struct OnlyCurrentTraitsOpaque {
    #[primary_span]
    pub span: Span,
}
#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_foreign)]
pub(crate) struct OnlyCurrentTraitsForeign {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_name)]
pub(crate) struct OnlyCurrentTraitsName<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_pointer)]
pub(crate) struct OnlyCurrentTraitsPointer<'a> {
    #[primary_span]
    pub span: Span,
    pub pointer: Ty<'a>,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_ty)]
pub(crate) struct OnlyCurrentTraitsTy<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_adt)]
pub(crate) struct OnlyCurrentTraitsAdt {
    #[primary_span]
    pub span: Span,
    pub name: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_analysis_only_current_traits_pointer_sugg,
    applicability = "maybe-incorrect"
)]
pub(crate) struct OnlyCurrentTraitsPointerSugg<'a> {
    #[suggestion_part(code = "WrapperType")]
    pub wrapper_span: Span,
    #[suggestion_part(code = "struct WrapperType(*{mut_key}{ptr_ty});\n\n")]
    pub(crate) struct_span: Span,
    pub mut_key: &'a str,
    pub ptr_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_not_supported_delegation)]
pub(crate) struct UnsupportedDelegation<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
    #[label]
    pub callee_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_method_should_return_future)]
pub(crate) struct MethodShouldReturnFuture {
    #[primary_span]
    pub span: Span,
    pub method_name: Symbol,
    #[note]
    pub trait_item_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_unused_generic_parameter)]
pub(crate) struct UnusedGenericParameter {
    #[primary_span]
    #[label]
    pub span: Span,
    pub param_name: Ident,
    pub param_def_kind: &'static str,
    #[label(hir_analysis_usage_spans)]
    pub usage_spans: Vec<Span>,
    #[subdiagnostic]
    pub help: UnusedGenericParameterHelp,
    #[help(hir_analysis_const_param_help)]
    pub const_param_help: bool,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_recursive_generic_parameter)]
pub(crate) struct RecursiveGenericParameter {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label]
    pub param_span: Span,
    pub param_name: Ident,
    pub param_def_kind: &'static str,
    #[subdiagnostic]
    pub help: UnusedGenericParameterHelp,
    #[note]
    pub note: (),
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedGenericParameterHelp {
    #[help(hir_analysis_unused_generic_parameter_adt_help)]
    Adt { param_name: Ident, phantom_data: String },
    #[help(hir_analysis_unused_generic_parameter_adt_no_phantom_data_help)]
    AdtNoPhantomData { param_name: Ident },
    #[help(hir_analysis_unused_generic_parameter_ty_alias_help)]
    TyAlias { param_name: Ident },
}

#[derive(Diagnostic)]
#[diag(hir_analysis_unconstrained_generic_parameter)]
pub(crate) struct UnconstrainedGenericParameter {
    #[primary_span]
    #[label]
    pub span: Span,
    pub param_name: Symbol,
    pub param_def_kind: &'static str,
    #[note(hir_analysis_const_param_note)]
    pub const_param_note: bool,
    #[note(hir_analysis_const_param_note2)]
    pub const_param_note2: bool,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_opaque_captures_higher_ranked_lifetime, code = E0657)]
pub(crate) struct OpaqueCapturesHigherRankedLifetime {
    #[primary_span]
    pub span: Span,
    #[label]
    pub label: Option<Span>,
    #[note]
    pub decl_span: Span,
    pub bad_place: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_pattern_type_non_const_range)]
pub(crate) struct NonConstRange {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidReceiverTyHint {
    #[note(hir_analysis_invalid_receiver_ty_help_weak_note)]
    Weak,
    #[note(hir_analysis_invalid_receiver_ty_help_nonnull_note)]
    NonNull,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_invalid_receiver_ty_no_arbitrary_self_types, code = E0307)]
#[note]
#[help(hir_analysis_invalid_receiver_ty_help_no_arbitrary_self_types)]
pub(crate) struct InvalidReceiverTyNoArbitrarySelfTypes<'tcx> {
    #[primary_span]
    pub span: Span,
    pub receiver_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_invalid_receiver_ty, code = E0307)]
#[note]
#[help(hir_analysis_invalid_receiver_ty_help)]
pub(crate) struct InvalidReceiverTy<'tcx> {
    #[primary_span]
    pub span: Span,
    pub receiver_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub hint: Option<InvalidReceiverTyHint>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_invalid_generic_receiver_ty, code = E0801)]
#[note]
#[help(hir_analysis_invalid_generic_receiver_ty_help)]
pub(crate) struct InvalidGenericReceiverTy<'tcx> {
    #[primary_span]
    pub span: Span,
    pub receiver_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cmse_inputs_stack_spill, code = E0798)]
#[note]
pub(crate) struct CmseInputsStackSpill {
    #[primary_span]
    #[label]
    pub span: Span,
    pub plural: bool,
    pub abi_name: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cmse_output_stack_spill, code = E0798)]
#[note(hir_analysis_note1)]
#[note(hir_analysis_note2)]
pub(crate) struct CmseOutputStackSpill {
    #[primary_span]
    #[label]
    pub span: Span,
    pub abi_name: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cmse_call_generic, code = E0798)]
pub(crate) struct CmseCallGeneric {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_bad_return_type_notation_position)]
pub(crate) struct BadReturnTypeNotation {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cmse_entry_generic, code = E0798)]
pub(crate) struct CmseEntryGeneric {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_register_type_unstable)]
pub(crate) struct RegisterTypeUnstable<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
}
