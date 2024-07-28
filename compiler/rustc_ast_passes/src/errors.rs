//! Errors emitted by ast_passes.

use rustc_ast::ParamKindOrd;
use rustc_errors::{
    codes::*, Applicability, Diag, EmissionGuarantee, SubdiagMessageOp, Subdiagnostic,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{symbol::Ident, Span, Symbol};

use crate::fluent_generated as fluent;

#[derive(Diagnostic)]
#[diag(ast_passes_visibility_not_permitted, code = E0449)]
pub struct VisibilityNotPermitted {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub note: VisibilityNotPermittedNote,
    #[suggestion(
        ast_passes_remove_qualifier_sugg,
        code = "",
        applicability = "machine-applicable"
    )]
    pub remove_qualifier_sugg: Span,
}

#[derive(Subdiagnostic)]
pub enum VisibilityNotPermittedNote {
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
pub struct TraitFnConst {
    #[primary_span]
    #[label]
    pub span: Span,
    pub in_impl: bool,
    #[label(ast_passes_const_context_label)]
    pub const_context_label: Option<Span>,
    #[suggestion(ast_passes_remove_const_sugg, code = "")]
    pub remove_const_sugg: (Span, Applicability),
    pub requires_multiple_changes: bool,
    #[suggestion(
        ast_passes_make_impl_const_sugg,
        code = "const ",
        applicability = "maybe-incorrect"
    )]
    pub make_impl_const_sugg: Option<Span>,
    #[suggestion(
        ast_passes_make_trait_const_sugg,
        code = "#[const_trait]\n",
        applicability = "maybe-incorrect"
    )]
    pub make_trait_const_sugg: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_bound)]
pub struct ForbiddenBound {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_const_param)]
pub struct ForbiddenConstParam {
    #[primary_span]
    pub const_param_spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_too_many)]
pub struct FnParamTooMany {
    #[primary_span]
    pub span: Span,
    pub max_num_args: usize,
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
#[diag(ast_passes_assoc_type_without_body)]
pub struct AssocTypeWithoutBody {
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
#[diag(ast_passes_extern_invalid_safety)]
pub struct InvalidSafetyOnExtern {
    #[primary_span]
    pub item_span: Span,
    #[suggestion(code = "unsafe ", applicability = "machine-applicable", style = "verbose")]
    pub block: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_item_invalid_safety)]
pub struct InvalidSafetyOnItem {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_bare_fn_invalid_safety)]
pub struct InvalidSafetyOnBareFn {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_unsafe_static)]
pub struct UnsafeStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_bound_in_context)]
pub struct BoundInContext<'a> {
    #[primary_span]
    pub span: Span,
    pub ctx: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_extern_types_cannot)]
#[note(ast_passes_extern_keyword_link)]
pub struct ExternTypesCannotHave<'a> {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    pub descr: &'a str,
    pub remove_descr: &'a str,
    #[label]
    pub block_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_body_in_extern)]
#[note(ast_passes_extern_keyword_link)]
pub struct BodyInExtern<'a> {
    #[primary_span]
    #[label(ast_passes_cannot_have)]
    pub span: Span,
    #[label(ast_passes_invalid)]
    pub body: Span,
    #[label(ast_passes_existing)]
    pub block: Span,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_body_extern)]
#[help]
#[note(ast_passes_extern_keyword_link)]
pub struct FnBodyInExtern {
    #[primary_span]
    #[label(ast_passes_cannot_have)]
    pub span: Span,
    #[suggestion(code = ";", applicability = "maybe-incorrect")]
    pub body: Span,
    #[label]
    pub block: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_extern_fn_qualifiers)]
pub struct FnQualifierInExtern {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    #[label]
    pub block: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_extern_item_ascii)]
#[note]
pub struct ExternItemAscii {
    #[primary_span]
    pub span: Span,
    #[label]
    pub block: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_bad_c_variadic)]
pub struct BadCVariadic {
    #[primary_span]
    pub span: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_item_underscore)]
pub struct ItemUnderscore<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nomangle_ascii, code = E0754)]
pub struct NoMangleAscii {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_module_nonascii, code = E0754)]
#[help]
pub struct ModuleNonAscii {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_generic, code = E0567)]
pub struct AutoTraitGeneric {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[label]
    pub ident: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_super_lifetime, code = E0568)]
pub struct AutoTraitBounds {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[label]
    pub ident: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_items, code = E0380)]
pub struct AutoTraitItems {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub total: Span,
    #[label]
    pub ident: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_generic_before_constraints)]
pub struct ArgsBeforeConstraint {
    #[primary_span]
    pub arg_spans: Vec<Span>,
    #[label(ast_passes_constraints)]
    pub constraints: Span,
    #[label(ast_passes_args)]
    pub args: Span,
    #[suggestion(code = "{suggestion}", applicability = "machine-applicable", style = "verbose")]
    pub data: Span,
    pub suggestion: String,
    pub constraint_len: usize,
    pub args_len: usize,
    #[subdiagnostic]
    pub constraint_spans: EmptyLabelManySpans,
    #[subdiagnostic]
    pub arg_spans2: EmptyLabelManySpans,
}

pub struct EmptyLabelManySpans(pub Vec<Span>);

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
pub struct PatternFnPointer {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_trait_object_single_bound, code = E0226)]
pub struct TraitObjectBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_impl_trait_path, code = E0667)]
pub struct ImplTraitPath {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nested_impl_trait, code = E0666)]
pub struct NestedImplTrait {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes_outer)]
    pub outer: Span,
    #[label(ast_passes_inner)]
    pub inner: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_at_least_one_trait)]
pub struct AtLeastOneTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_out_of_order_params)]
pub struct OutOfOrderParams<'a> {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "{ordered_params}", applicability = "machine-applicable")]
    pub sugg_span: Span,
    pub param_ord: &'a ParamKindOrd,
    pub max_param: &'a ParamKindOrd,
    pub ordered_params: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_obsolete_auto)]
#[help]
pub struct ObsoleteAuto {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_unsafe_negative_impl, code = E0198)]
pub struct UnsafeNegativeImpl {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes_negative)]
    pub negative: Span,
    #[label(ast_passes_unsafe)]
    pub r#unsafe: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_inherent_cannot_be)]
pub struct InherentImplCannot<'a> {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes_because)]
    pub annotation_span: Span,
    pub annotation: &'a str,
    #[label(ast_passes_type)]
    pub self_ty: Span,
    #[note(ast_passes_only_trait)]
    pub only_trait: Option<()>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_inherent_cannot_be, code = E0197)]
pub struct InherentImplCannotUnsafe<'a> {
    #[primary_span]
    pub span: Span,
    #[label(ast_passes_because)]
    pub annotation_span: Span,
    pub annotation: &'a str,
    #[label(ast_passes_type)]
    pub self_ty: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_unsafe_item)]
pub struct UnsafeItem {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_missing_unsafe_on_extern)]
pub struct MissingUnsafeOnExtern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fieldless_union)]
pub struct FieldlessUnion {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_where_clause_after_type_alias)]
#[note]
pub struct WhereClauseAfterTypeAlias {
    #[primary_span]
    pub span: Span,
    #[help]
    pub help: Option<()>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_where_clause_before_type_alias)]
#[note]
pub struct WhereClauseBeforeTypeAlias {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: WhereClauseBeforeTypeAliasSugg,
}

#[derive(Subdiagnostic)]

pub enum WhereClauseBeforeTypeAliasSugg {
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
pub struct GenericDefaultTrailing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nested_lifetimes, code = E0316)]
pub struct NestedLifetimes {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_optional_trait_supertrait)]
#[note]
pub struct OptionalTraitSupertrait {
    #[primary_span]
    pub span: Span,
    pub path_str: String,
}

#[derive(Diagnostic)]
#[diag(ast_passes_optional_trait_object)]
pub struct OptionalTraitObject {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_bound_trait_object)]
pub struct ConstBoundTraitObject {
    #[primary_span]
    pub span: Span,
}

// FIXME(effects): Consider making the note/reason the message of the diagnostic.
// FIXME(effects): Provide structured suggestions (e.g., add `const` / `#[const_trait]` here).
#[derive(Diagnostic)]
#[diag(ast_passes_tilde_const_disallowed)]
pub struct TildeConstDisallowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub reason: TildeConstReason,
}

#[derive(Subdiagnostic, Copy, Clone)]
pub enum TildeConstReason {
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
#[diag(ast_passes_const_and_async)]
pub struct ConstAndAsync {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label(ast_passes_const)]
    pub cspan: Span,
    #[label(ast_passes_async)]
    pub aspan: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_and_c_variadic)]
pub struct ConstAndCVariadic {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label(ast_passes_const)]
    pub const_span: Span,
    #[label(ast_passes_variadic)]
    pub variadic_spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_pattern_in_foreign, code = E0130)]
// FIXME: deduplicate with rustc_lint (`BuiltinLintDiag::PatternsInFnsWithoutBody`)
pub struct PatternInForeign {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_pattern_in_bodiless, code = E0642)]
// FIXME: deduplicate with rustc_lint (`BuiltinLintDiag::PatternsInFnsWithoutBody`)
pub struct PatternInBodiless {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_equality_in_where)]
#[note]
pub struct EqualityInWhere {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub assoc: Option<AssociatedSuggestion>,
    #[subdiagnostic]
    pub assoc2: Option<AssociatedSuggestion2>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    ast_passes_suggestion,
    code = "{param}: {path}",
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub struct AssociatedSuggestion {
    #[primary_span]
    pub span: Span,
    pub ident: Ident,
    pub param: Ident,
    pub path: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(ast_passes_suggestion_path, applicability = "maybe-incorrect")]
pub struct AssociatedSuggestion2 {
    #[suggestion_part(code = "{args}")]
    pub span: Span,
    pub args: String,
    #[suggestion_part(code = "")]
    pub predicate: Span,
    pub trait_segment: Ident,
    pub potential_assoc: Ident,
}

#[derive(Diagnostic)]
#[diag(ast_passes_stability_outside_std, code = E0734)]
pub struct StabilityOutsideStd {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_feature_on_non_nightly, code = E0554)]
pub struct FeatureOnNonNightly {
    #[primary_span]
    pub span: Span,
    pub channel: &'static str,
    #[subdiagnostic]
    pub stable_features: Vec<StableFeature>,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub sugg: Option<Span>,
}

pub struct StableFeature {
    pub name: Symbol,
    pub since: Symbol,
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
pub struct IncompatibleFeatures {
    #[primary_span]
    pub spans: Vec<Span>,
    pub f1: Symbol,
    pub f2: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_show_span)]
pub struct ShowSpan {
    #[primary_span]
    pub span: Span,
    pub msg: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_negative_bound_not_supported)]
pub struct NegativeBoundUnsupported {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_constraint_on_negative_bound)]
pub struct ConstraintOnNegativeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_negative_bound_with_parenthetical_notation)]
pub struct NegativeBoundWithParentheticalNotation {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_unnamed_field_ty)]
pub struct InvalidUnnamedFieldTy {
    #[primary_span]
    pub span: Span,
    #[label]
    pub ty_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_unnamed_field)]
pub struct InvalidUnnamedField {
    #[primary_span]
    pub span: Span,
    #[label]
    pub ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_anon_struct_or_union_not_allowed)]
pub struct AnonStructOrUnionNotAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    pub struct_or_union: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_match_arm_with_no_body)]
pub struct MatchArmWithNoBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " => todo!(),", applicability = "has-placeholders")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_precise_capturing_not_allowed_here)]
pub struct PreciseCapturingNotAllowedHere {
    #[primary_span]
    pub span: Span,
    pub loc: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_precise_capturing_duplicated)]
pub struct DuplicatePreciseCapturing {
    #[primary_span]
    pub bound1: Span,
    #[label]
    pub bound2: Span,
}
