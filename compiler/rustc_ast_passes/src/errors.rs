//! Errors emitted by ast_passes.

use rustc_ast::ParamKindOrd;
use rustc_errors::{
    codes::*, Applicability, Diag, EmissionGuarantee, SubdiagMessageOp, Subdiagnostic,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{symbol::Ident, Span, Symbol};

use crate::fluent_generated as fluent;

#[derive(Diagnostic)]
#[diag(ast_passes_keyword_lifetime)]
pub(crate) struct KeywordLifetime {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_label)]
pub(crate) struct InvalidLabel {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_visibility_not_permitted, code = E0449)]
pub(crate) struct VisibilityNotPermitted {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) note: VisibilityNotPermittedNote,
    #[suggestion(
        ast_passes_remove_qualifier_sugg,
        code = "",
        applicability = "machine-applicable"
    )]
    pub(crate) remove_qualifier_sugg: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum VisibilityNotPermittedNote {
    #[note(ast_passes_enum_variant)]
    EnumVariant,
    #[note(ast_passes_trait_impl)]
    TraitImpl,
    #[note(ast_passes_individual_impl_items)]
    IndividualImplItems,
    #[note(ast_passes_individual_foreign_items)]
    IndividualForeignItems,
}

#[derive(Diagnostic)]
#[diag(ast_passes_trait_fn_const, code = E0379)]
pub(crate) struct TraitFnConst {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) in_impl: bool,
    #[label(ast_passes_const_context_label)]
    pub(crate) const_context_label: Option<Span>,
    #[suggestion(ast_passes_remove_const_sugg, code = "")]
    pub(crate) remove_const_sugg: (Span, Applicability),
    pub(crate) requires_multiple_changes: bool,
    #[suggestion(
        ast_passes_make_impl_const_sugg,
        code = "const ",
        applicability = "maybe-incorrect"
    )]
    pub(crate) make_impl_const_sugg: Option<Span>,
    #[suggestion(
        ast_passes_make_trait_const_sugg,
        code = "#[const_trait]\n",
        applicability = "maybe-incorrect"
    )]
    pub(crate) make_trait_const_sugg: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_bound)]
pub(crate) struct ForbiddenBound {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_too_many)]
pub(crate) struct FnParamTooMany {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) max_num_args: usize,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_c_var_args_not_last)]
pub(crate) struct FnParamCVarArgsNotLast {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_doc_comment)]
pub(crate) struct FnParamDocComment {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_forbidden_attr)]
pub(crate) struct FnParamForbiddenAttr {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_forbidden_self)]
#[note]
pub(crate) struct FnParamForbiddenSelf {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_default)]
pub(crate) struct ForbiddenDefault {
    #[primary_span]
    pub(crate) span: Span,
    #[label]
    pub(crate) def_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_assoc_const_without_body)]
pub(crate) struct AssocConstWithoutBody {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub(crate) replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_assoc_fn_without_body)]
pub(crate) struct AssocFnWithoutBody {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = " {{ <body> }}", applicability = "has-placeholders")]
    pub(crate) replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_assoc_type_without_body)]
pub(crate) struct AssocTypeWithoutBody {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = " = <type>;", applicability = "has-placeholders")]
    pub(crate) replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_without_body)]
pub(crate) struct ConstWithoutBody {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub(crate) replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_static_without_body)]
pub(crate) struct StaticWithoutBody {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub(crate) replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_ty_alias_without_body)]
pub(crate) struct TyAliasWithoutBody {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = " = <type>;", applicability = "has-placeholders")]
    pub(crate) replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_without_body)]
pub(crate) struct FnWithoutBody {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = " {{ <body> }}", applicability = "has-placeholders")]
    pub(crate) replace_span: Span,
    #[subdiagnostic]
    pub(crate) extern_block_suggestion: Option<ExternBlockSuggestion>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ExternBlockSuggestion {
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
#[diag(ast_passes_extern_invalid_safety)]
pub struct InvalidSafetyOnExtern {
    #[primary_span]
    pub item_span: Span,
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub block: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_bound_in_context)]
pub(crate) struct BoundInContext<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ctx: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_extern_types_cannot)]
#[note(ast_passes_extern_keyword_link)]
pub(crate) struct ExternTypesCannotHave<'a> {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub(crate) span: Span,
    pub(crate) descr: &'a str,
    pub(crate) remove_descr: &'a str,
    #[label]
    pub(crate) block_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_body_in_extern)]
#[note(ast_passes_extern_keyword_link)]
pub(crate) struct BodyInExtern<'a> {
    #[primary_span]
    #[label(ast_passes_cannot_have)]
    pub(crate) span: Span,
    #[label(ast_passes_invalid)]
    pub(crate) body: Span,
    #[label(ast_passes_existing)]
    pub(crate) block: Span,
    pub(crate) kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_body_extern)]
#[help]
#[note(ast_passes_extern_keyword_link)]
pub(crate) struct FnBodyInExtern {
    #[primary_span]
    #[label(ast_passes_cannot_have)]
    pub(crate) span: Span,
    #[suggestion(code = ";", applicability = "maybe-incorrect")]
    pub(crate) body: Span,
    #[label]
    pub(crate) block: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_extern_fn_qualifiers)]
pub(crate) struct FnQualifierInExtern {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub(crate) span: Span,
    #[label]
    pub(crate) block: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_extern_item_ascii)]
#[note]
pub(crate) struct ExternItemAscii {
    #[primary_span]
    pub(crate) span: Span,
    #[label]
    pub(crate) block: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_bad_c_variadic)]
pub(crate) struct BadCVariadic {
    #[primary_span]
    pub(crate) span: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_item_underscore)]
pub(crate) struct ItemUnderscore<'a> {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nomangle_ascii, code = E0754)]
pub(crate) struct NoMangleAscii {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_module_nonascii, code = E0754)]
#[help]
pub(crate) struct ModuleNonAscii {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_generic, code = E0567)]
pub(crate) struct AutoTraitGeneric {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub(crate) span: Span,
    #[label]
    pub(crate) ident: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_super_lifetime, code = E0568)]
pub(crate) struct AutoTraitBounds {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub(crate) span: Span,
    #[label]
    pub(crate) ident: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_items, code = E0380)]
pub(crate) struct AutoTraitItems {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub(crate) total: Span,
    #[label]
    pub(crate) ident: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_generic_before_constraints)]
pub(crate) struct ArgsBeforeConstraint {
    #[primary_span]
    pub(crate) arg_spans: Vec<Span>,
    #[label(ast_passes_constraints)]
    pub(crate) constraints: Span,
    #[label(ast_passes_args)]
    pub(crate) args: Span,
    #[suggestion(code = "{suggestion}", applicability = "machine-applicable", style = "verbose")]
    pub(crate) data: Span,
    pub(crate) suggestion: String,
    pub(crate) constraint_len: usize,
    pub(crate) args_len: usize,
    #[subdiagnostic]
    pub(crate) constraint_spans: EmptyLabelManySpans,
    #[subdiagnostic]
    pub(crate) arg_spans2: EmptyLabelManySpans,
}

pub(crate) struct EmptyLabelManySpans(pub(crate) Vec<Span>);

// The derive for `Vec<Span>` does multiple calls to `span_label`, adding commas between each
impl Subdiagnostic for EmptyLabelManySpans {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _: &F,
    ) {
        diag.span_labels(self.0, "");
    }
}

#[derive(Diagnostic)]
#[diag(ast_passes_pattern_in_fn_pointer, code = E0561)]
pub(crate) struct PatternFnPointer {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_trait_object_single_bound, code = E0226)]
pub(crate) struct TraitObjectBound {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_impl_trait_path, code = E0667)]
pub(crate) struct ImplTraitPath {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nested_impl_trait, code = E0666)]
pub(crate) struct NestedImplTrait {
    #[primary_span]
    pub(crate) span: Span,
    #[label(ast_passes_outer)]
    pub(crate) outer: Span,
    #[label(ast_passes_inner)]
    pub(crate) inner: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_at_least_one_trait)]
pub(crate) struct AtLeastOneTrait {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_out_of_order_params)]
pub(crate) struct OutOfOrderParams<'a> {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
    #[suggestion(code = "{ordered_params}", applicability = "machine-applicable")]
    pub(crate) sugg_span: Span,
    pub(crate) param_ord: &'a ParamKindOrd,
    pub(crate) max_param: &'a ParamKindOrd,
    pub(crate) ordered_params: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_obsolete_auto)]
#[help]
pub(crate) struct ObsoleteAuto {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_unsafe_negative_impl, code = E0198)]
pub(crate) struct UnsafeNegativeImpl {
    #[primary_span]
    pub(crate) span: Span,
    #[label(ast_passes_negative)]
    pub(crate) negative: Span,
    #[label(ast_passes_unsafe)]
    pub(crate) r#unsafe: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_inherent_cannot_be)]
pub(crate) struct InherentImplCannot<'a> {
    #[primary_span]
    pub(crate) span: Span,
    #[label(ast_passes_because)]
    pub(crate) annotation_span: Span,
    pub(crate) annotation: &'a str,
    #[label(ast_passes_type)]
    pub(crate) self_ty: Span,
    #[note(ast_passes_only_trait)]
    pub(crate) only_trait: Option<()>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_inherent_cannot_be, code = E0197)]
pub(crate) struct InherentImplCannotUnsafe<'a> {
    #[primary_span]
    pub(crate) span: Span,
    #[label(ast_passes_because)]
    pub(crate) annotation_span: Span,
    pub(crate) annotation: &'a str,
    #[label(ast_passes_type)]
    pub(crate) self_ty: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_unsafe_item)]
pub(crate) struct UnsafeItem {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) kind: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_missing_unsafe_on_extern)]
pub struct MissingUnsafeOnExtern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fieldless_union)]
pub(crate) struct FieldlessUnion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_where_clause_after_type_alias)]
#[note]
pub(crate) struct WhereClauseAfterTypeAlias {
    #[primary_span]
    pub(crate) span: Span,
    #[help]
    pub(crate) help: Option<()>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_where_clause_before_type_alias)]
#[note]
pub(crate) struct WhereClauseBeforeTypeAlias {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) sugg: WhereClauseBeforeTypeAliasSugg,
}

#[derive(Subdiagnostic)]

pub(crate) enum WhereClauseBeforeTypeAliasSugg {
    #[suggestion(ast_passes_remove_suggestion, applicability = "machine-applicable", code = "")]
    Remove {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        ast_passes_move_suggestion,
        applicability = "machine-applicable",
        style = "verbose"
    )]
    Move {
        #[suggestion_part(code = "")]
        left: Span,
        snippet: String,
        #[suggestion_part(code = "{snippet}")]
        right: Span,
    },
}

#[derive(Diagnostic)]
#[diag(ast_passes_generic_default_trailing)]
pub(crate) struct GenericDefaultTrailing {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nested_lifetimes, code = E0316)]
pub(crate) struct NestedLifetimes {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_optional_trait_supertrait)]
#[note]
pub(crate) struct OptionalTraitSupertrait {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) path_str: String,
}

#[derive(Diagnostic)]
#[diag(ast_passes_optional_trait_object)]
pub(crate) struct OptionalTraitObject {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_bound_trait_object)]
pub(crate) struct ConstBoundTraitObject {
    #[primary_span]
    pub(crate) span: Span,
}

// FIXME(effects): Consider making the note/reason the message of the diagnostic.
// FIXME(effects): Provide structured suggestions (e.g., add `const` / `#[const_trait]` here).
#[derive(Diagnostic)]
#[diag(ast_passes_tilde_const_disallowed)]
pub(crate) struct TildeConstDisallowed {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) reason: TildeConstReason,
}

#[derive(Subdiagnostic)]
pub(crate) enum TildeConstReason {
    #[note(ast_passes_closure)]
    Closure,
    #[note(ast_passes_function)]
    Function {
        #[primary_span]
        ident: Span,
    },
    #[note(ast_passes_trait)]
    Trait {
        #[primary_span]
        span: Span,
    },
    #[note(ast_passes_trait_impl)]
    TraitImpl {
        #[primary_span]
        span: Span,
    },
    #[note(ast_passes_impl)]
    Impl {
        #[primary_span]
        span: Span,
    },
    #[note(ast_passes_trait_assoc_ty)]
    TraitAssocTy {
        #[primary_span]
        span: Span,
    },
    #[note(ast_passes_trait_impl_assoc_ty)]
    TraitImplAssocTy {
        #[primary_span]
        span: Span,
    },
    #[note(ast_passes_inherent_assoc_ty)]
    InherentAssocTy {
        #[primary_span]
        span: Span,
    },
    #[note(ast_passes_object)]
    TraitObject,
    #[note(ast_passes_item)]
    Item,
}

#[derive(Diagnostic)]
#[diag(ast_passes_incompatible_trait_bound_modifiers)]
pub(crate) struct IncompatibleTraitBoundModifiers {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) left: &'static str,
    pub(crate) right: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_and_async)]
pub(crate) struct ConstAndAsync {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
    #[label(ast_passes_const)]
    pub(crate) cspan: Span,
    #[label(ast_passes_async)]
    pub(crate) aspan: Span,
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_and_c_variadic)]
pub(crate) struct ConstAndCVariadic {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
    #[label(ast_passes_const)]
    pub(crate) const_span: Span,
    #[label(ast_passes_variadic)]
    pub(crate) variadic_spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_pattern_in_foreign, code = E0130)]
// FIXME: deduplicate with rustc_lint (`BuiltinLintDiag::PatternsInFnsWithoutBody`)
pub(crate) struct PatternInForeign {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_pattern_in_bodiless, code = E0642)]
// FIXME: deduplicate with rustc_lint (`BuiltinLintDiag::PatternsInFnsWithoutBody`)
pub(crate) struct PatternInBodiless {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_equality_in_where)]
#[note]
pub(crate) struct EqualityInWhere {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) assoc: Option<AssociatedSuggestion>,
    #[subdiagnostic]
    pub(crate) assoc2: Option<AssociatedSuggestion2>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    ast_passes_suggestion,
    code = "{param}: {path}",
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedSuggestion {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
    pub(crate) param: Ident,
    pub(crate) path: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(ast_passes_suggestion_path, applicability = "maybe-incorrect")]
pub(crate) struct AssociatedSuggestion2 {
    #[suggestion_part(code = "{args}")]
    pub(crate) span: Span,
    pub(crate) args: String,
    #[suggestion_part(code = "")]
    pub(crate) predicate: Span,
    pub(crate) trait_segment: Ident,
    pub(crate) potential_assoc: Ident,
}

#[derive(Diagnostic)]
#[diag(ast_passes_stability_outside_std, code = E0734)]
pub(crate) struct StabilityOutsideStd {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_feature_on_non_nightly, code = E0554)]
pub(crate) struct FeatureOnNonNightly {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) channel: &'static str,
    #[subdiagnostic]
    pub(crate) stable_features: Vec<StableFeature>,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub(crate) sugg: Option<Span>,
}

pub(crate) struct StableFeature {
    pub(crate) name: Symbol,
    pub(crate) since: Symbol,
}

impl Subdiagnostic for StableFeature {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _: &F,
    ) {
        diag.arg("name", self.name);
        diag.arg("since", self.since);
        diag.help(fluent::ast_passes_stable_since);
    }
}

#[derive(Diagnostic)]
#[diag(ast_passes_incompatible_features)]
#[help]
pub(crate) struct IncompatibleFeatures {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
    pub(crate) f1: Symbol,
    pub(crate) f2: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_show_span)]
pub(crate) struct ShowSpan {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) msg: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_negative_bound_not_supported)]
pub(crate) struct NegativeBoundUnsupported {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_constraint_on_negative_bound)]
pub(crate) struct ConstraintOnNegativeBound {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_negative_bound_with_parenthetical_notation)]
pub(crate) struct NegativeBoundWithParentheticalNotation {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_unnamed_field_ty)]
pub(crate) struct InvalidUnnamedFieldTy {
    #[primary_span]
    pub(crate) span: Span,
    #[label]
    pub(crate) ty_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_unnamed_field)]
pub(crate) struct InvalidUnnamedField {
    #[primary_span]
    pub(crate) span: Span,
    #[label]
    pub(crate) ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_anon_struct_or_union_not_allowed)]
pub(crate) struct AnonStructOrUnionNotAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) struct_or_union: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_match_arm_with_no_body)]
pub(crate) struct MatchArmWithNoBody {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = " => todo!(),", applicability = "has-placeholders")]
    pub(crate) suggestion: Span,
}
