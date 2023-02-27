//! Errors emitted by `rustc_hir_analysis`.

use crate::fluent_generated as fluent;
use rustc_errors::{
    error_code, Applicability, DiagnosticBuilder, ErrorGuaranteed, Handler, IntoDiagnostic,
    MultiSpan,
};
use rustc_macros::Diagnostic;
use rustc_middle::ty::Ty;
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag(hir_analysis_unrecognized_atomic_operation, code = "E0092")]
pub struct UnrecognizedAtomicOperation<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub op: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_wrong_number_of_generic_arguments_to_intrinsic, code = "E0094")]
pub struct WrongNumberOfGenericArgumentsToIntrinsic<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub found: usize,
    pub expected: usize,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_unrecognized_intrinsic_function, code = "E0093")]
pub struct UnrecognizedIntrinsicFunction {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_lifetimes_or_bounds_mismatch_on_trait, code = "E0195")]
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
#[diag(hir_analysis_async_trait_impl_should_be_async)]
pub struct AsyncTraitImplShouldBeAsync {
    #[primary_span]
    // #[label]
    pub span: Span,
    #[label(hir_analysis_trait_item_label)]
    pub trait_item_span: Option<Span>,
    pub method_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_drop_impl_on_wrong_item, code = "E0120")]
pub struct DropImplOnWrongItem {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_field_already_declared, code = "E0124")]
pub struct FieldAlreadyDeclared {
    pub field_name: Ident,
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_analysis_previous_decl_label)]
    pub prev_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_copy_impl_on_type_with_dtor, code = "E0184")]
pub struct CopyImplOnTypeWithDtor {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_multiple_relaxed_default_bounds, code = "E0203")]
pub struct MultipleRelaxedDefaultBounds {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_copy_impl_on_non_adt, code = "E0206")]
pub struct CopyImplOnNonAdt {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_trait_object_declared_with_no_traits, code = "E0224")]
pub struct TraitObjectDeclaredWithNoTraits {
    #[primary_span]
    pub span: Span,
    #[label(hir_analysis_alias_span)]
    pub trait_alias_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_ambiguous_lifetime_bound, code = "E0227")]
pub struct AmbiguousLifetimeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_assoc_type_binding_not_allowed, code = "E0229")]
pub struct AssocTypeBindingNotAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_typeof_reserved_keyword_used, code = "E0516")]
pub struct TypeofReservedKeywordUsed<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion(style = "verbose", code = "{ty}")]
    pub opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_value_of_associated_struct_already_specified, code = "E0719")]
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

pub struct MissingTypeParams {
    pub span: Span,
    pub def_span: Span,
    pub span_snippet: Option<String>,
    pub missing_type_params: Vec<Symbol>,
    pub empty_generic_args: bool,
}

// Manual implementation of `IntoDiagnostic` to be able to call `span_to_snippet`.
impl<'a> IntoDiagnostic<'a> for MissingTypeParams {
    #[track_caller]
    fn into_diagnostic(self, handler: &'a Handler) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        let mut err = handler.struct_span_err_with_code(
            self.span,
            fluent::hir_analysis_missing_type_params,
            error_code!(E0393),
        );
        err.set_arg("parameterCount", self.missing_type_params.len());
        err.set_arg(
            "parameters",
            self.missing_type_params
                .iter()
                .map(|n| format!("`{}`", n))
                .collect::<Vec<_>>()
                .join(", "),
        );

        err.span_label(self.def_span, fluent::hir_analysis_label);

        let mut suggested = false;
        // Don't suggest setting the type params if there are some already: the order is
        // tricky to get right and the user will already know what the syntax is.
        if let Some(snippet) = self.span_snippet && self.empty_generic_args {
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
#[diag(hir_analysis_manual_implementation, code = "E0183")]
#[help]
pub struct ManualImplementation {
    #[primary_span]
    #[label]
    pub span: Span,
    pub trait_name: String,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_substs_on_overridden_impl)]
pub struct SubstsOnOverriddenImpl {
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
#[diag(hir_analysis_linkage_type, code = "E0791")]
pub(crate) struct LinkageType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[help]
#[diag(hir_analysis_auto_deref_reached_recursion_limit, code = "E0055")]
pub struct AutoDerefReachedRecursionLimit<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub ty: Ty<'a>,
    pub suggested_limit: rustc_session::Limit,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_where_clause_on_main, code = "E0646")]
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
#[diag(hir_analysis_start_not_track_caller)]
pub(crate) struct StartTrackCaller {
    #[primary_span]
    pub span: Span,
    #[label]
    pub start: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_start_not_async, code = "E0752")]
pub(crate) struct StartAsync {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_start_function_where, code = "E0647")]
pub(crate) struct StartFunctionWhere {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_start_function_parameters, code = "E0132")]
pub(crate) struct StartFunctionParameters {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_main_function_return_type_generic, code = "E0131")]
pub(crate) struct MainFunctionReturnTypeGeneric {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_main_function_async, code = "E0752")]
pub(crate) struct MainFunctionAsync {
    #[primary_span]
    pub span: Span,
    #[label]
    pub asyncness: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_main_function_generic_parameters, code = "E0131")]
pub(crate) struct MainFunctionGenericParameters {
    #[primary_span]
    pub span: Span,
    #[label]
    pub label_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis_variadic_function_compatible_convention, code = "E0045")]
pub(crate) struct VariadicFunctionCompatibleConvention<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub conventions: &'a str,
}
