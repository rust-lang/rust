//! Errors emitted by ast_passes.

use rustc_ast::ParamKindOrd;
use rustc_errors::AddToDiagnostic;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{symbol::Ident, Span, Symbol};

use crate::fluent_generated as fluent;

#[derive(Diagnostic)]
#[diag(ast_passes_keyword_lifetime)]
#[must_use]
pub struct KeywordLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_label)]
#[must_use]
pub struct InvalidLabel {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_visibility_not_permitted, code = "E0449")]
#[must_use]
pub struct VisibilityNotPermitted {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub note: VisibilityNotPermittedNote,
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
#[diag(ast_passes_trait_fn_const, code = "E0379")]
#[must_use]
pub struct TraitFnConst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_lifetime_bound)]
#[must_use]
pub struct ForbiddenLifetimeBound {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_too_many)]
#[must_use]
pub struct FnParamTooMany {
    #[primary_span]
    pub span: Span,
    pub max_num_args: usize,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_c_var_args_only)]
#[must_use]
pub struct FnParamCVarArgsOnly {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_c_var_args_not_last)]
#[must_use]
pub struct FnParamCVarArgsNotLast {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_doc_comment)]
#[must_use]
pub struct FnParamDocComment {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_forbidden_attr)]
#[must_use]
pub struct FnParamForbiddenAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_param_forbidden_self)]
#[note]
#[must_use]
pub struct FnParamForbiddenSelf {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_forbidden_default)]
#[must_use]
pub struct ForbiddenDefault {
    #[primary_span]
    pub span: Span,
    #[label]
    pub def_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_assoc_const_without_body)]
#[must_use]
pub struct AssocConstWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_assoc_fn_without_body)]
#[must_use]
pub struct AssocFnWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " {{ <body> }}", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_assoc_type_without_body)]
#[must_use]
pub struct AssocTypeWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <type>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_without_body)]
#[must_use]
pub struct ConstWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_static_without_body)]
#[must_use]
pub struct StaticWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <expr>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_ty_alias_without_body)]
#[must_use]
pub struct TyAliasWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = <type>;", applicability = "has-placeholders")]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fn_without_body)]
#[must_use]
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
#[diag(ast_passes_bound_in_context)]
#[must_use]
pub struct BoundInContext<'a> {
    #[primary_span]
    pub span: Span,
    pub ctx: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_extern_types_cannot)]
#[note(ast_passes_extern_keyword_link)]
#[must_use]
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
#[must_use]
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
#[must_use]
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
#[must_use]
pub struct FnQualifierInExtern {
    #[primary_span]
    pub span: Span,
    #[label]
    pub block: Span,
    #[suggestion(code = "fn ", applicability = "maybe-incorrect", style = "verbose")]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_extern_item_ascii)]
#[note]
#[must_use]
pub struct ExternItemAscii {
    #[primary_span]
    pub span: Span,
    #[label]
    pub block: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_bad_c_variadic)]
#[must_use]
pub struct BadCVariadic {
    #[primary_span]
    pub span: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_item_underscore)]
#[must_use]
pub struct ItemUnderscore<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nomangle_ascii, code = "E0754")]
#[must_use]
pub struct NoMangleAscii {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_module_nonascii, code = "E0754")]
#[help]
#[must_use]
pub struct ModuleNonAscii {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_generic, code = "E0567")]
#[must_use]
pub struct AutoTraitGeneric {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[label]
    pub ident: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_super_lifetime, code = "E0568")]
#[must_use]
pub struct AutoTraitBounds {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[label]
    pub ident: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_auto_items, code = "E0380")]
#[must_use]
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
#[must_use]
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
impl AddToDiagnostic for EmptyLabelManySpans {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        diag.span_labels(self.0, "");
    }
}

#[derive(Diagnostic)]
#[diag(ast_passes_pattern_in_fn_pointer, code = "E0561")]
#[must_use]
pub struct PatternFnPointer {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_trait_object_single_bound, code = "E0226")]
#[must_use]
pub struct TraitObjectBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_impl_trait_path, code = "E0667")]
#[must_use]
pub struct ImplTraitPath {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nested_impl_trait, code = "E0666")]
#[must_use]
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
#[must_use]
pub struct AtLeastOneTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_out_of_order_params)]
#[must_use]
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
#[must_use]
pub struct ObsoleteAuto {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_unsafe_negative_impl, code = "E0198")]
#[must_use]
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
#[must_use]
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
#[diag(ast_passes_inherent_cannot_be, code = "E0197")]
#[must_use]
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
#[must_use]
pub struct UnsafeItem {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_fieldless_union)]
#[must_use]
pub struct FieldlessUnion {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_where_clause_after_type_alias)]
#[note]
#[must_use]
pub struct WhereClauseAfterTypeAlias {
    #[primary_span]
    pub span: Span,
    #[help]
    pub help: Option<()>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_where_clause_before_type_alias)]
#[note]
#[must_use]
pub struct WhereClauseBeforeTypeAlias {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: WhereClauseBeforeTypeAliasSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    ast_passes_suggestion,
    applicability = "machine-applicable",
    style = "verbose"
)]
pub struct WhereClauseBeforeTypeAliasSugg {
    #[suggestion_part(code = "")]
    pub left: Span,
    pub snippet: String,
    #[suggestion_part(code = "{snippet}")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_generic_default_trailing)]
#[must_use]
pub struct GenericDefaultTrailing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_nested_lifetimes, code = "E0316")]
#[must_use]
pub struct NestedLifetimes {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_optional_trait_supertrait)]
#[note]
#[must_use]
pub struct OptionalTraitSupertrait {
    #[primary_span]
    pub span: Span,
    pub path_str: String,
}

#[derive(Diagnostic)]
#[diag(ast_passes_optional_trait_object)]
#[must_use]
pub struct OptionalTraitObject {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_tilde_const_disallowed)]
#[must_use]
pub struct TildeConstDisallowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub reason: TildeConstReason,
}

#[derive(Subdiagnostic)]
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
    #[note(ast_passes_object)]
    TraitObject,
    #[note(ast_passes_item)]
    Item,
}

#[derive(Diagnostic)]
#[diag(ast_passes_optional_const_exclusive)]
#[must_use]
pub struct OptionalConstExclusive {
    #[primary_span]
    pub span: Span,
    pub modifier: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_const_and_async)]
#[must_use]
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
#[must_use]
pub struct ConstAndCVariadic {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label(ast_passes_const)]
    pub const_span: Span,
    #[label(ast_passes_variadic)]
    pub variadic_spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_passes_pattern_in_foreign, code = "E0130")]
#[must_use]
pub struct PatternInForeign {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_pattern_in_bodiless, code = "E0642")]
#[must_use]
pub struct PatternInBodiless {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_equality_in_where)]
#[note]
#[must_use]
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
#[diag(ast_passes_stability_outside_std, code = "E0734")]
#[must_use]
pub struct StabilityOutsideStd {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_feature_on_non_nightly, code = "E0554")]
#[must_use]
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

impl AddToDiagnostic for StableFeature {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        diag.set_arg("name", self.name);
        diag.set_arg("since", self.since);
        diag.help(fluent::ast_passes_stable_since);
    }
}

#[derive(Diagnostic)]
#[diag(ast_passes_incompatible_features)]
#[help]
#[must_use]
pub struct IncompatibleFeatures {
    #[primary_span]
    pub spans: Vec<Span>,
    pub f1: Symbol,
    pub f2: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_passes_show_span)]
#[must_use]
pub struct ShowSpan {
    #[primary_span]
    pub span: Span,
    pub msg: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_negative_bound_not_supported)]
#[must_use]
pub struct NegativeBoundUnsupported {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_constraint_on_negative_bound)]
#[must_use]
pub struct ConstraintOnNegativeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_unnamed_field_ty)]
#[must_use]
pub struct InvalidUnnamedFieldTy {
    #[primary_span]
    pub span: Span,
    #[label]
    pub ty_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_invalid_unnamed_field)]
#[must_use]
pub struct InvalidUnnamedField {
    #[primary_span]
    pub span: Span,
    #[label]
    pub ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_passes_anon_struct_or_union_not_allowed)]
#[must_use]
pub struct AnonStructOrUnionNotAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    pub struct_or_union: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_passes_match_arm_with_no_body)]
#[must_use]
pub struct MatchArmWithNoBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " => todo!(),", applicability = "has-placeholders")]
    pub suggestion: Span,
}
