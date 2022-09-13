use rustc_errors::{Applicability, MultiSpan};
use rustc_macros::{LintDiagnostic, SessionDiagnostic, SessionSubdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(LintDiagnostic)]
#[diag(passes::outer_crate_level_attr)]
pub struct OuterCrateLevelAttr;

#[derive(LintDiagnostic)]
#[diag(passes::inner_crate_level_attr)]
pub struct InnerCrateLevelAttr;

#[derive(LintDiagnostic)]
#[diag(passes::ignored_attr_with_macro)]
pub struct IgnoredAttrWithMacro<'a> {
    pub sym: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes::ignored_attr)]
pub struct IgnoredAttr<'a> {
    pub sym: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes::inline_ignored_function_prototype)]
pub struct IgnoredInlineAttrFnProto;

#[derive(LintDiagnostic)]
#[diag(passes::inline_ignored_constants)]
#[warning]
#[note]
pub struct IgnoredInlineAttrConstants;

#[derive(SessionDiagnostic)]
#[diag(passes::inline_not_fn_or_closure, code = "E0518")]
pub struct InlineNotFnOrClosure {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::no_coverage_ignored_function_prototype)]
pub struct IgnoredNoCoverageFnProto;

#[derive(LintDiagnostic)]
#[diag(passes::no_coverage_propagate)]
pub struct IgnoredNoCoveragePropagate;

#[derive(LintDiagnostic)]
#[diag(passes::no_coverage_fn_defn)]
pub struct IgnoredNoCoverageFnDefn;

#[derive(SessionDiagnostic)]
#[diag(passes::no_coverage_not_coverable, code = "E0788")]
pub struct IgnoredNoCoverageNotCoverable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::should_be_applied_to_fn)]
pub struct AttrShouldBeAppliedToFn {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::naked_tracked_caller, code = "E0736")]
pub struct NakedTrackedCaller {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::should_be_applied_to_fn, code = "E0739")]
pub struct TrackedCallerWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::should_be_applied_to_struct_enum, code = "E0701")]
pub struct NonExhaustiveWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::should_be_applied_to_trait)]
pub struct AttrShouldBeAppliedToTrait {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::target_feature_on_statement)]
pub struct TargetFeatureOnStatement;

#[derive(SessionDiagnostic)]
#[diag(passes::should_be_applied_to_static)]
pub struct AttrShouldBeAppliedToStatic {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_expect_str)]
pub struct DocExpectStr<'a> {
    #[primary_span]
    pub attr_span: Span,
    pub attr_name: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_alias_empty)]
pub struct DocAliasEmpty<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_alias_bad_char)]
pub struct DocAliasBadChar<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub char_: char,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_alias_start_end)]
pub struct DocAliasStartEnd<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_alias_bad_location)]
pub struct DocAliasBadLocation<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub location: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_alias_not_an_alias)]
pub struct DocAliasNotAnAlias<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes::doc_alias_duplicated)]
pub struct DocAliasDuplicated {
    #[label]
    pub first_defn: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_alias_not_string_literal)]
pub struct DocAliasNotStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_alias_malformed)]
pub struct DocAliasMalformed {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_keyword_empty_mod)]
pub struct DocKeywordEmptyMod {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_keyword_not_mod)]
pub struct DocKeywordNotMod {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_keyword_invalid_ident)]
pub struct DocKeywordInvalidIdent {
    #[primary_span]
    pub span: Span,
    pub doc_keyword: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_fake_variadic_not_valid)]
pub struct DocFakeVariadicNotValid {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_keyword_only_impl)]
pub struct DocKeywordOnlyImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_inline_conflict)]
#[help]
pub struct DocKeywordConflict {
    #[primary_span]
    pub spans: MultiSpan,
}

#[derive(LintDiagnostic)]
#[diag(passes::doc_inline_only_use)]
#[note]
pub struct DocInlineOnlyUse {
    #[label]
    pub attr_span: Span,
    #[label(passes::not_a_use_item_label)]
    pub item_span: Option<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(passes::doc_attr_not_crate_level)]
pub struct DocAttrNotCrateLevel<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_name: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes::doc_test_unknown)]
pub struct DocTestUnknown {
    pub path: String,
}

#[derive(LintDiagnostic)]
#[diag(passes::doc_test_takes_list)]
pub struct DocTestTakesList;

#[derive(LintDiagnostic)]
#[diag(passes::doc_primitive)]
pub struct DocPrimitive;

#[derive(LintDiagnostic)]
#[diag(passes::doc_test_unknown_any)]
pub struct DocTestUnknownAny {
    pub path: String,
}

#[derive(LintDiagnostic)]
#[diag(passes::doc_test_unknown_spotlight)]
#[note]
#[note(passes::no_op_note)]
pub struct DocTestUnknownSpotlight {
    pub path: String,
    #[suggestion_short(applicability = "machine-applicable", code = "notable_trait")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::doc_test_unknown_include)]
pub struct DocTestUnknownInclude {
    pub path: String,
    pub value: String,
    pub inner: &'static str,
    #[suggestion(code = "#{inner}[doc = include_str!(\"{value}\")]")]
    pub sugg: (Span, Applicability),
}

#[derive(LintDiagnostic)]
#[diag(passes::doc_invalid)]
pub struct DocInvalid;

#[derive(SessionDiagnostic)]
#[diag(passes::pass_by_value)]
pub struct PassByValue {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::allow_incoherent_impl)]
pub struct AllowIncoherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::has_incoherent_inherent_impl)]
pub struct HasIncoherentInherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::must_use_async)]
pub struct MustUseAsync {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::must_use_no_effect)]
pub struct MustUseNoEffect {
    pub article: &'static str,
    pub target: rustc_hir::Target,
}

#[derive(SessionDiagnostic)]
#[diag(passes::must_not_suspend)]
pub struct MustNotSuspend {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::cold)]
#[warning]
pub struct Cold {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::link)]
#[warning]
pub struct Link {
    #[label]
    pub span: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(passes::link_name)]
#[warning]
pub struct LinkName<'a> {
    #[help]
    pub attr_span: Option<Span>,
    #[label]
    pub span: Span,
    pub value: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(passes::no_link)]
pub struct NoLink {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::export_name)]
pub struct ExportName {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_layout_scalar_valid_range_not_struct)]
pub struct RustcLayoutScalarValidRangeNotStruct {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_layout_scalar_valid_range_arg)]
pub struct RustcLayoutScalarValidRangeArg {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_legacy_const_generics_only)]
pub struct RustcLegacyConstGenericsOnly {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub param_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_legacy_const_generics_index)]
pub struct RustcLegacyConstGenericsIndex {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub generics_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_legacy_const_generics_index_exceed)]
pub struct RustcLegacyConstGenericsIndexExceed {
    #[primary_span]
    #[label]
    pub span: Span,
    pub arg_count: usize,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_legacy_const_generics_index_negative)]
pub struct RustcLegacyConstGenericsIndexNegative {
    #[primary_span]
    pub invalid_args: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_dirty_clean)]
pub struct RustcDirtyClean {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::link_section)]
#[warning]
pub struct LinkSection {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::no_mangle_foreign)]
#[warning]
#[note]
pub struct NoMangleForeign {
    #[label]
    pub span: Span,
    #[suggestion(applicability = "machine-applicable")]
    pub attr_span: Span,
    pub foreign_item_kind: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(passes::no_mangle)]
#[warning]
pub struct NoMangle {
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::repr_ident, code = "E0565")]
pub struct ReprIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::repr_conflicting, code = "E0566")]
pub struct ReprConflicting;

#[derive(SessionDiagnostic)]
#[diag(passes::used_static)]
pub struct UsedStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::used_compiler_linker)]
pub struct UsedCompilerLinker {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(passes::allow_internal_unstable)]
pub struct AllowInternalUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::debug_visualizer_placement)]
pub struct DebugVisualizerPlacement {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::debug_visualizer_invalid)]
#[note(passes::note_1)]
#[note(passes::note_2)]
#[note(passes::note_3)]
pub struct DebugVisualizerInvalid {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_allow_const_fn_unstable)]
pub struct RustcAllowConstFnUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_std_internal_symbol)]
pub struct RustcStdInternalSymbol {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::const_trait)]
pub struct ConstTrait {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::link_ordinal)]
pub struct LinkOrdinal {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::stability_promotable)]
pub struct StabilityPromotable {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::deprecated)]
pub struct Deprecated;

#[derive(LintDiagnostic)]
#[diag(passes::macro_use)]
pub struct MacroUse {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(passes::macro_export)]
pub struct MacroExport;

#[derive(LintDiagnostic)]
#[diag(passes::plugin_registrar)]
pub struct PluginRegistrar;

#[derive(SessionSubdiagnostic)]
pub enum UnusedNote {
    #[note(passes::unused_empty_lints_note)]
    EmptyList { name: Symbol },
    #[note(passes::unused_no_lints_note)]
    NoLints { name: Symbol },
    #[note(passes::unused_default_method_body_const_note)]
    DefaultMethodBodyConst,
}

#[derive(LintDiagnostic)]
#[diag(passes::unused)]
pub struct Unused {
    #[suggestion(applicability = "machine-applicable")]
    pub attr_span: Span,
    #[subdiagnostic]
    pub note: UnusedNote,
}

#[derive(SessionDiagnostic)]
#[diag(passes::non_exported_macro_invalid_attrs, code = "E0518")]
pub struct NonExportedMacroInvalidAttrs {
    #[primary_span]
    #[label]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::unused_duplicate)]
pub struct UnusedDuplicate {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    #[warning]
    pub warning: Option<()>,
}

#[derive(SessionDiagnostic)]
#[diag(passes::unused_multiple)]
pub struct UnusedMultiple {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_lint_opt_ty)]
pub struct RustcLintOptTy {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::rustc_lint_opt_deny_field_access)]
pub struct RustcLintOptDenyFieldAccess {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(passes::collapse_debuginfo)]
pub struct CollapseDebuginfo {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}
