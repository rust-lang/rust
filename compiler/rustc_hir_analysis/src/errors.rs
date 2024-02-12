//! Errors emitted by `rustc_hir_analysis`.

use crate::fluent_generated as fluent;
use rustc_errors::{
    codes::*, Applicability, DiagCtxt, DiagnosticBuilder, EmissionGuarantee, IntoDiagnostic, Level,
    MultiSpan,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag(hir_analysis_ambiguous_assoc_item)]
pub struct AmbiguousAssocItem<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub assoc_kind: &'static str,
    pub assoc_name: Ident,
    pub ty_param_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_assoc_kind_mismatch)]
pub struct AssocKindMismatch {
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
pub struct AssocKindMismatchWrapInBracesSugg {
    #[suggestion_part(code = "{{ ")]
    pub lo: Span,
    #[suggestion_part(code = " }}")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_assoc_item_not_found, code = E0220)]
pub struct AssocItemNotFound<'a> {
    #[primary_span]
    pub span: Span,
    pub assoc_name: Ident,
    pub assoc_kind: &'static str,
    pub ty_param_name: &'a str,
    #[subdiagnostic]
    pub label: Option<AssocItemNotFoundLabel<'a>>,
    #[subdiagnostic]
    pub sugg: Option<AssocItemNotFoundSugg<'a>>,
}

#[derive(Subdiagnostic)]
pub enum AssocItemNotFoundLabel<'a> {
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

pub enum AssocItemNotFoundSugg<'a> {
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
    #[suggestion(hir_analysis_assoc_item_not_found_other_sugg, code = "{suggested_name}")]
    Other {
        #[primary_span]
        span: Span,
        #[applicability]
        applicability: Applicability,
        ty_param_name: &'a str,
        assoc_kind: &'static str,
        suggested_name: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag(hir_analysis_unrecognized_atomic_operation, code = E0092)]
pub struct UnrecognizedAtomicOperation<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub op: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_wrong_number_of_generic_arguments_to_intrinsic, code = E0094)]
pub struct WrongNumberOfGenericArgumentsToIntrinsic<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub found: usize,
    pub expected: usize,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_unrecognized_intrinsic_function, code = E0093)]
pub struct UnrecognizedIntrinsicFunction {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_lifetimes_or_bounds_mismatch_on_trait, code = E0195)]
pub struct LifetimesOrBoundsMismatchOnTrait {
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
pub struct DropImplOnWrongItem {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
pub enum FieldAlreadyDeclared {
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
pub struct FieldAlreadyDeclaredNestedHelp {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_copy_impl_on_type_with_dtor, code = E0184)]
pub struct CopyImplOnTypeWithDtor {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_multiple_relaxed_default_bounds, code = E0203)]
pub struct MultipleRelaxedDefaultBounds {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_copy_impl_on_non_adt, code = E0206)]
pub struct CopyImplOnNonAdt {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_const_param_ty_impl_on_non_adt)]
pub struct ConstParamTyImplOnNonAdt {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_trait_object_declared_with_no_traits, code = E0224)]
pub struct TraitObjectDeclaredWithNoTraits {
    #[primary_span]
    pub span: Span,
    #[label(hir_analysis_alias_span)]
    pub trait_alias_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_ambiguous_lifetime_bound, code = E0227)]
pub struct AmbiguousLifetimeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_assoc_type_binding_not_allowed, code = E0229)]
pub struct AssocTypeBindingNotAllowed {
    #[primary_span]
    #[label]
    pub span: Span,

    #[subdiagnostic]
    pub fn_trait_expansion: Option<ParenthesizedFnTraitExpansion>,
}

#[derive(Subdiagnostic)]
#[help(hir_analysis_parenthesized_fn_trait_expansion)]
pub struct ParenthesizedFnTraitExpansion {
    #[primary_span]
    pub span: Span,

    pub expanded_type: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_typeof_reserved_keyword_used, code = E0516)]
pub struct TypeofReservedKeywordUsed<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion(style = "verbose", code = "{ty}")]
    pub opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_value_of_associated_struct_already_specified, code = E0719)]
pub struct ValueOfAssociatedStructAlreadySpecified {
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
pub struct UnconstrainedOpaqueType {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub what: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_tait_forward_compat)]
#[note]
pub struct TaitForwardCompat {
    #[primary_span]
    pub span: Span,
    #[note]
    pub item_span: Span,
}

pub struct MissingTypeParams {
    pub span: Span,
    pub def_span: Span,
    pub span_snippet: Option<String>,
    pub missing_type_params: Vec<Symbol>,
    pub empty_generic_args: bool,
}

// Manual implementation of `IntoDiagnostic` to be able to call `span_to_snippet`.
impl<'a, G: EmissionGuarantee> IntoDiagnostic<'a, G> for MissingTypeParams {
    #[track_caller]
    fn into_diagnostic(self, dcx: &'a DiagCtxt, level: Level) -> DiagnosticBuilder<'a, G> {
        let mut err = DiagnosticBuilder::new(dcx, level, fluent::hir_analysis_missing_type_params);
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
                err.span_suggestion(
                    self.span,
                    fluent::hir_analysis_suggestion,
                    format!(
                        "{}<{}>",
                        snippet,
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
pub struct ManualImplementation {
    #[primary_span]
    #[label]
    pub span: Span,
    pub trait_name: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_generic_args_on_overridden_impl)]
pub struct GenericArgsOnOverriddenImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_const_impl_for_non_const_trait)]
pub struct ConstImplForNonConstTrait {
    #[primary_span]
    pub trait_ref_span: Span,
    pub trait_name: String,
    #[suggestion(applicability = "machine-applicable", code = "#[const_trait]")]
    pub local_trait_span: Option<Span>,
    #[note]
    pub marking: (),
    #[note(hir_analysis_adding)]
    pub adding: (),
}

#[derive(Diagnostic)]
#[diag(hir_analysis_const_bound_for_non_const_trait)]
pub struct ConstBoundForNonConstTrait {
    #[primary_span]
    pub span: Span,
    pub modifier: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_self_in_impl_self)]
pub struct SelfInImplSelf {
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
pub struct AutoDerefReachedRecursionLimit<'a> {
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
pub(crate) struct VariadicFunctionCompatibleConvention<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub conventions: &'a str,
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
    pub variances_of: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_type_of)]
pub(crate) struct TypeOf<'tcx> {
    #[primary_span]
    pub span: Span,
    pub type_of: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_pass_to_variadic_function, code = E0617)]
pub(crate) struct PassToVariadicFunction<'tcx, 'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub cast_ty: &'a str,
    #[suggestion(code = "{replace}", applicability = "machine-applicable")]
    pub sugg_span: Option<Span>,
    pub replace: String,
    #[help]
    pub help: Option<()>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cast_thin_pointer_to_fat_pointer, code = E0607)]
pub(crate) struct CastThinPointerToFatPointer<'tcx> {
    #[primary_span]
    pub span: Span,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: String,
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
pub(crate) struct AssociatedTypeTraitUninferredGenericParams {
    #[primary_span]
    pub span: Span,
    #[suggestion(style = "verbose", applicability = "maybe-incorrect", code = "{bound}")]
    pub inferred_sugg: Option<Span>,
    pub bound: String,
    #[subdiagnostic]
    pub mpart_sugg: Option<AssociatedTypeTraitUninferredGenericParamsMultipartSuggestion>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_analysis_associated_type_trait_uninferred_generic_params_multipart_suggestion,
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedTypeTraitUninferredGenericParamsMultipartSuggestion {
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
pub enum ImplNotMarkedDefault {
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
#[diag(hir_analysis_const_specialize)]
pub(crate) struct ConstSpecialize {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_static_specialize)]
pub(crate) struct StaticSpecialize {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_missing_tilde_const)]
pub(crate) struct MissingTildeConst {
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
pub struct UnusedAssociatedTypeBounds {
    #[suggestion(code = "")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(hir_analysis_rpitit_refined)]
#[note]
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

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_ty_outside, code = E0390)]
#[help]
pub struct InherentTyOutside {
    #[primary_span]
    #[help(hir_analysis_span_help)]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0378)]
pub struct DispatchFromDynCoercion<'a> {
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
pub struct DispatchFromDynRepr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_ty_outside_relevant, code = E0390)]
#[help]
pub struct InherentTyOutsideRelevant {
    #[primary_span]
    pub span: Span,
    #[help(hir_analysis_span_help)]
    pub help_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_ty_outside_new, code = E0116)]
#[note]
pub struct InherentTyOutsideNew {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_ty_outside_primitive, code = E0390)]
#[help]
pub struct InherentTyOutsidePrimitive {
    #[primary_span]
    pub span: Span,
    #[help(hir_analysis_span_help)]
    pub help_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_primitive_ty, code = E0390)]
#[help]
pub struct InherentPrimitiveTy<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub note: Option<InherentPrimitiveTyNote<'a>>,
}

#[derive(Subdiagnostic)]
#[note(hir_analysis_inherent_primitive_ty_note)]
pub struct InherentPrimitiveTyNote<'a> {
    pub subty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_dyn, code = E0785)]
#[note]
pub struct InherentDyn {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_inherent_nominal, code = E0118)]
#[note]
pub struct InherentNominal {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_dispatch_from_dyn_zst, code = E0378)]
#[note]
pub struct DispatchFromDynZST<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0378)]
pub struct DispatchFromDynSingle<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
    #[note(hir_analysis_coercion_between_struct_single_note)]
    pub note: bool,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_dispatch_from_dyn_multi, code = E0378)]
#[note]
pub struct DispatchFromDynMulti {
    #[primary_span]
    pub span: Span,
    #[note(hir_analysis_coercions_note)]
    pub coercions_note: bool,
    pub number: usize,
    pub coercions: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0376)]
pub struct DispatchFromDynStruct<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_may, code = E0377)]
pub struct DispatchFromDynSame<'a> {
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
pub struct CoerceUnsizedOneField<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
    #[note(hir_analysis_coercion_between_struct_single_note)]
    pub note: bool,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_coerce_unsized_multi, code = E0375)]
#[note]
pub struct CoerceUnsizedMulti {
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
pub struct CoerceUnsizedMay<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_trait_cannot_impl_for_ty, code = E0204)]
pub struct TraitCannotImplForTy {
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
pub struct ImplForTyRequires {
    #[primary_span]
    pub span: MultiSpan,
    pub error_predicate: String,
    pub trait_name: String,
    pub ty: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_traits_with_defualt_impl, code = E0321)]
#[note]
pub struct TraitsWithDefaultImpl<'a> {
    #[primary_span]
    pub span: Span,
    pub traits: String,
    pub problematic_kind: &'a str,
    pub self_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cross_crate_traits, code = E0321)]
pub struct CrossCrateTraits<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub traits: String,
    pub self_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_cross_crate_traits_defined, code = E0321)]
pub struct CrossCrateTraitsDefined {
    #[primary_span]
    #[label]
    pub span: Span,
    pub traits: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_ty_param_first_local, code = E0210)]
#[note]
pub struct TyParamFirstLocal<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note(hir_analysis_case_note)]
    pub note: (),
    pub param_ty: Ty<'a>,
    pub local_type: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_ty_param_some, code = E0210)]
#[note]
pub struct TyParamSome<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note(hir_analysis_only_note)]
    pub note: (),
    pub param_ty: Ty<'a>,
}

#[derive(Diagnostic)]
pub enum OnlyCurrentTraits<'a> {
    #[diag(hir_analysis_only_current_traits_outside, code = E0117)]
    Outside {
        #[primary_span]
        #[label(hir_analysis_only_current_traits_label)]
        span: Span,
        #[note(hir_analysis_only_current_traits_note)]
        note: (),
        #[subdiagnostic]
        opaque: Vec<OnlyCurrentTraitsOpaque>,
        #[subdiagnostic]
        foreign: Vec<OnlyCurrentTraitsForeign>,
        #[subdiagnostic]
        name: Vec<OnlyCurrentTraitsName<'a>>,
        #[subdiagnostic]
        pointer: Vec<OnlyCurrentTraitsPointer<'a>>,
        #[subdiagnostic]
        ty: Vec<OnlyCurrentTraitsTy<'a>>,
        #[subdiagnostic]
        sugg: Option<OnlyCurrentTraitsPointerSugg<'a>>,
    },
    #[diag(hir_analysis_only_current_traits_primitive, code = E0117)]
    Primitive {
        #[primary_span]
        #[label(hir_analysis_only_current_traits_label)]
        span: Span,
        #[note(hir_analysis_only_current_traits_note)]
        note: (),
        #[subdiagnostic]
        opaque: Vec<OnlyCurrentTraitsOpaque>,
        #[subdiagnostic]
        foreign: Vec<OnlyCurrentTraitsForeign>,
        #[subdiagnostic]
        name: Vec<OnlyCurrentTraitsName<'a>>,
        #[subdiagnostic]
        pointer: Vec<OnlyCurrentTraitsPointer<'a>>,
        #[subdiagnostic]
        ty: Vec<OnlyCurrentTraitsTy<'a>>,
        #[subdiagnostic]
        sugg: Option<OnlyCurrentTraitsPointerSugg<'a>>,
    },
    #[diag(hir_analysis_only_current_traits_arbitrary, code = E0117)]
    Arbitrary {
        #[primary_span]
        #[label(hir_analysis_only_current_traits_label)]
        span: Span,
        #[note(hir_analysis_only_current_traits_note)]
        note: (),
        #[subdiagnostic]
        opaque: Vec<OnlyCurrentTraitsOpaque>,
        #[subdiagnostic]
        foreign: Vec<OnlyCurrentTraitsForeign>,
        #[subdiagnostic]
        name: Vec<OnlyCurrentTraitsName<'a>>,
        #[subdiagnostic]
        pointer: Vec<OnlyCurrentTraitsPointer<'a>>,
        #[subdiagnostic]
        ty: Vec<OnlyCurrentTraitsTy<'a>>,
        #[subdiagnostic]
        sugg: Option<OnlyCurrentTraitsPointerSugg<'a>>,
    },
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_opaque)]
pub struct OnlyCurrentTraitsOpaque {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_foreign)]
pub struct OnlyCurrentTraitsForeign {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_name)]
pub struct OnlyCurrentTraitsName<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_pointer)]
pub struct OnlyCurrentTraitsPointer<'a> {
    #[primary_span]
    pub span: Span,
    pub pointer: Ty<'a>,
}

#[derive(Subdiagnostic)]
#[label(hir_analysis_only_current_traits_ty)]
pub struct OnlyCurrentTraitsTy<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_analysis_only_current_traits_pointer_sugg,
    applicability = "maybe-incorrect"
)]
pub struct OnlyCurrentTraitsPointerSugg<'a> {
    #[suggestion_part(code = "WrapperType")]
    pub wrapper_span: Span,
    #[suggestion_part(code = "struct WrapperType(*{mut_key}{ptr_ty});\n\n")]
    pub struct_span: Span,
    pub mut_key: &'a str,
    pub ptr_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_static_mut_ref, code = E0796)]
#[note]
pub struct StaticMutRef {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: StaticMutRefSugg,
}

#[derive(Subdiagnostic)]
pub enum StaticMutRefSugg {
    #[suggestion(
        hir_analysis_suggestion,
        style = "verbose",
        code = "addr_of!({var})",
        applicability = "maybe-incorrect"
    )]
    Shared {
        #[primary_span]
        span: Span,
        var: String,
    },
    #[suggestion(
        hir_analysis_suggestion_mut,
        style = "verbose",
        code = "addr_of_mut!({var})",
        applicability = "maybe-incorrect"
    )]
    Mut {
        #[primary_span]
        span: Span,
        var: String,
    },
}

// STATIC_MUT_REF lint
#[derive(LintDiagnostic)]
#[diag(hir_analysis_static_mut_ref_lint)]
#[note]
pub struct RefOfMutStatic<'a> {
    pub shared: &'a str,
    #[note(hir_analysis_why_note)]
    pub why_note: (),
    #[subdiagnostic]
    pub label: RefOfMutStaticLabel,
    #[subdiagnostic]
    pub sugg: RefOfMutStaticSugg,
}

#[derive(Subdiagnostic)]
pub enum RefOfMutStaticLabel {
    #[label(hir_analysis_label)]
    Shared {
        #[primary_span]
        span: Span,
    },
    #[label(hir_analysis_label_mut)]
    Mut {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub enum RefOfMutStaticSugg {
    #[suggestion(
        hir_analysis_suggestion,
        style = "verbose",
        code = "addr_of!({var})",
        applicability = "maybe-incorrect"
    )]
    Shared {
        #[primary_span]
        span: Span,
        var: String,
    },
    #[suggestion(
        hir_analysis_suggestion_mut,
        style = "verbose",
        code = "addr_of_mut!({var})",
        applicability = "maybe-incorrect"
    )]
    Mut {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Diagnostic)]
#[diag(hir_analysis_not_supported_delegation)]
pub struct NotSupportedDelegation<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
    #[label]
    pub callee_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_method_should_return_future)]
pub struct MethodShouldReturnFuture {
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
    #[subdiagnostic]
    pub help: UnusedGenericParameterHelp,
    #[help(hir_analysis_const_param_help)]
    pub const_param_help: Option<()>,
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
pub enum UnnamedFieldsRepr<'a> {
    #[diag(hir_analysis_unnamed_fields_repr_missing_repr_c)]
    MissingReprC {
        #[primary_span]
        #[label]
        span: Span,
        adt_kind: &'static str,
        adt_name: Symbol,
        #[subdiagnostic]
        unnamed_fields: Vec<UnnamedFieldsReprFieldDefined>,
        #[suggestion(code = "#[repr(C)]\n")]
        sugg_span: Span,
    },
    #[diag(hir_analysis_unnamed_fields_repr_field_missing_repr_c)]
    FieldMissingReprC {
        #[primary_span]
        #[label]
        span: Span,
        #[label(hir_analysis_field_ty_label)]
        field_ty_span: Span,
        field_ty: Ty<'a>,
        field_adt_kind: &'static str,
        #[suggestion(code = "#[repr(C)]\n")]
        sugg_span: Span,
    },
}

#[derive(Subdiagnostic)]
#[note(hir_analysis_unnamed_fields_repr_field_defined)]
pub struct UnnamedFieldsReprFieldDefined {
    #[primary_span]
    pub span: Span,
}
