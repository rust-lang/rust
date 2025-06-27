use std::io::Error;
use std::path::{Path, PathBuf};

use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, DiagSymbolList, Diagnostic, EmissionGuarantee, Level,
    MultiSpan, Subdiagnostic,
};
use rustc_hir::Target;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{MainDefinition, Ty};
use rustc_span::{DUMMY_SP, Span, Symbol};

use crate::check_attr::ProcMacroKind;
use crate::fluent_generated as fluent;
use crate::lang_items::Duplicate;

#[derive(LintDiagnostic)]
#[diag(passes_incorrect_do_not_recommend_location)]
pub(crate) struct IncorrectDoNotRecommendLocation;

#[derive(LintDiagnostic)]
#[diag(passes_incorrect_do_not_recommend_args)]
pub(crate) struct DoNotRecommendDoesNotExpectArgs;

#[derive(Diagnostic)]
#[diag(passes_autodiff_attr)]
pub(crate) struct AutoDiffAttr {
    #[primary_span]
    #[label]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_loop_match_attr)]
pub(crate) struct LoopMatchAttr {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub node_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_const_continue_attr)]
pub(crate) struct ConstContinueAttr {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub node_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_mixed_export_name_and_no_mangle)]
pub(crate) struct MixedExportNameAndNoMangle {
    #[label]
    #[suggestion(style = "verbose", code = "", applicability = "machine-applicable")]
    pub no_mangle_span: Span,
    #[note]
    pub export_name_span: Span,
    pub no_mangle_attr: &'static str,
    pub export_name_attr: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(passes_outer_crate_level_attr)]
pub(crate) struct OuterCrateLevelAttr;

#[derive(LintDiagnostic)]
#[diag(passes_inner_crate_level_attr)]
pub(crate) struct InnerCrateLevelAttr;

#[derive(LintDiagnostic)]
#[diag(passes_ignored_attr_with_macro)]
pub(crate) struct IgnoredAttrWithMacro<'a> {
    pub sym: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes_ignored_attr)]
pub(crate) struct IgnoredAttr<'a> {
    pub sym: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes_inline_ignored_function_prototype)]
pub(crate) struct IgnoredInlineAttrFnProto;

#[derive(LintDiagnostic)]
#[diag(passes_inline_ignored_constants)]
#[warning]
#[note]
pub(crate) struct IgnoredInlineAttrConstants;

#[derive(Diagnostic)]
#[diag(passes_inline_not_fn_or_closure, code = E0518)]
pub(crate) struct InlineNotFnOrClosure {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

/// "coverage attribute not allowed here"
#[derive(Diagnostic)]
#[diag(passes_coverage_attribute_not_allowed, code = E0788)]
pub(crate) struct CoverageAttributeNotAllowed {
    #[primary_span]
    pub attr_span: Span,
    /// "not a function, impl block, or module"
    #[label(passes_not_fn_impl_mod)]
    pub not_fn_impl_mod: Option<Span>,
    /// "function has no body"
    #[label(passes_no_body)]
    pub no_body: Option<Span>,
    /// "coverage attribute can be applied to a function (with body), impl block, or module"
    #[help]
    pub help: (),
}

#[derive(Diagnostic)]
#[diag(passes_optimize_invalid_target)]
pub(crate) struct OptimizeInvalidTarget {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
    pub on_crate: bool,
}

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_fn)]
pub(crate) struct AttrShouldBeAppliedToFn {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
    pub on_crate: bool,
}

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_fn, code = E0739)]
pub(crate) struct TrackedCallerWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
    pub on_crate: bool,
}

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_struct_enum, code = E0701)]
pub(crate) struct NonExhaustiveWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_non_exaustive_with_default_field_values)]
pub(crate) struct NonExhaustiveWithDefaultFieldValues {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_trait)]
pub(crate) struct AttrShouldBeAppliedToTrait {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_target_feature_on_statement)]
pub(crate) struct TargetFeatureOnStatement;

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_static)]
pub(crate) struct AttrShouldBeAppliedToStatic {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_expect_str)]
pub(crate) struct DocExpectStr<'a> {
    #[primary_span]
    pub attr_span: Span,
    pub attr_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_empty)]
pub(crate) struct DocAliasEmpty<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_bad_char)]
pub(crate) struct DocAliasBadChar<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub char_: char,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_start_end)]
pub(crate) struct DocAliasStartEnd<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_bad_location)]
pub(crate) struct DocAliasBadLocation<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub location: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_not_an_alias)]
pub(crate) struct DocAliasNotAnAlias<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_alias_duplicated)]
pub(crate) struct DocAliasDuplicated {
    #[label]
    pub first_defn: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_not_string_literal)]
pub(crate) struct DocAliasNotStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_malformed)]
pub(crate) struct DocAliasMalformed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_keyword_empty_mod)]
pub(crate) struct DocKeywordEmptyMod {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_keyword_not_keyword)]
#[help]
pub(crate) struct DocKeywordNotKeyword {
    #[primary_span]
    pub span: Span,
    pub keyword: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_doc_keyword_not_mod)]
pub(crate) struct DocKeywordNotMod {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_fake_variadic_not_valid)]
pub(crate) struct DocFakeVariadicNotValid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_keyword_only_impl)]
pub(crate) struct DocKeywordOnlyImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_search_unbox_invalid)]
pub(crate) struct DocSearchUnboxInvalid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_inline_conflict)]
#[help]
pub(crate) struct DocKeywordConflict {
    #[primary_span]
    pub spans: MultiSpan,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_inline_only_use)]
#[note]
pub(crate) struct DocInlineOnlyUse {
    #[label]
    pub attr_span: Span,
    #[label(passes_not_a_use_item_label)]
    pub item_span: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_masked_only_extern_crate)]
#[note]
pub(crate) struct DocMaskedOnlyExternCrate {
    #[label]
    pub attr_span: Span,
    #[label(passes_not_an_extern_crate_label)]
    pub item_span: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_masked_not_extern_crate_self)]
pub(crate) struct DocMaskedNotExternCrateSelf {
    #[label]
    pub attr_span: Span,
    #[label(passes_extern_crate_self_label)]
    pub item_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_doc_attr_not_crate_level)]
pub(crate) struct DocAttrNotCrateLevel<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_name: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown)]
pub(crate) struct DocTestUnknown {
    pub path: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_literal)]
pub(crate) struct DocTestLiteral;

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_takes_list)]
pub(crate) struct DocTestTakesList;

#[derive(LintDiagnostic)]
#[diag(passes_doc_cfg_hide_takes_list)]
pub(crate) struct DocCfgHideTakesList;

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown_any)]
pub(crate) struct DocTestUnknownAny {
    pub path: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown_spotlight)]
#[note]
#[note(passes_no_op_note)]
pub(crate) struct DocTestUnknownSpotlight {
    pub path: String,
    #[suggestion(style = "short", applicability = "machine-applicable", code = "notable_trait")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown_passes)]
#[note]
#[help]
#[note(passes_no_op_note)]
pub(crate) struct DocTestUnknownPasses {
    pub path: String,
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown_plugins)]
#[note]
#[note(passes_no_op_note)]
pub(crate) struct DocTestUnknownPlugins {
    pub path: String,
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown_include)]
pub(crate) struct DocTestUnknownInclude {
    pub path: String,
    pub value: String,
    pub inner: &'static str,
    #[suggestion(code = "#{inner}[doc = include_str!(\"{value}\")]")]
    pub sugg: (Span, Applicability),
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_invalid)]
pub(crate) struct DocInvalid;

#[derive(Diagnostic)]
#[diag(passes_pass_by_value)]
pub(crate) struct PassByValue {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_allow_incoherent_impl)]
pub(crate) struct AllowIncoherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_has_incoherent_inherent_impl)]
pub(crate) struct HasIncoherentInherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_both_ffi_const_and_pure, code = E0757)]
pub(crate) struct BothFfiConstAndPure {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_ffi_pure_invalid_target, code = E0755)]
pub(crate) struct FfiPureInvalidTarget {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_ffi_const_invalid_target, code = E0756)]
pub(crate) struct FfiConstInvalidTarget {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_must_use_no_effect)]
pub(crate) struct MustUseNoEffect {
    pub article: &'static str,
    pub target: rustc_hir::Target,
}

#[derive(Diagnostic)]
#[diag(passes_must_not_suspend)]
pub(crate) struct MustNotSuspend {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_cold)]
#[warning]
pub(crate) struct Cold {
    #[label]
    pub span: Span,
    pub on_crate: bool,
}

#[derive(LintDiagnostic)]
#[diag(passes_link)]
#[warning]
pub(crate) struct Link {
    #[label]
    pub span: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(passes_link_name)]
#[warning]
pub(crate) struct LinkName<'a> {
    #[help]
    pub attr_span: Option<Span>,
    #[label]
    pub span: Span,
    pub value: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_no_link)]
pub(crate) struct NoLink {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_export_name)]
pub(crate) struct ExportName {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_layout_scalar_valid_range_not_struct)]
pub(crate) struct RustcLayoutScalarValidRangeNotStruct {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_layout_scalar_valid_range_arg)]
pub(crate) struct RustcLayoutScalarValidRangeArg {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_legacy_const_generics_only)]
pub(crate) struct RustcLegacyConstGenericsOnly {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub param_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_legacy_const_generics_index)]
pub(crate) struct RustcLegacyConstGenericsIndex {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub generics_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_legacy_const_generics_index_exceed)]
pub(crate) struct RustcLegacyConstGenericsIndexExceed {
    #[primary_span]
    #[label]
    pub span: Span,
    pub arg_count: usize,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_legacy_const_generics_index_negative)]
pub(crate) struct RustcLegacyConstGenericsIndexNegative {
    #[primary_span]
    pub invalid_args: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_dirty_clean)]
pub(crate) struct RustcDirtyClean {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_link_section)]
#[warning]
pub(crate) struct LinkSection {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_no_mangle_foreign)]
#[warning]
#[note]
pub(crate) struct NoMangleForeign {
    #[label]
    pub span: Span,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub attr_span: Span,
    pub foreign_item_kind: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(passes_no_mangle)]
#[warning]
pub(crate) struct NoMangle {
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_repr_conflicting, code = E0566)]
pub(crate) struct ReprConflicting {
    #[primary_span]
    pub hint_spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_repr_align_greater_than_target_max, code = E0589)]
#[note]
pub(crate) struct InvalidReprAlignForTarget {
    #[primary_span]
    pub span: Span,
    pub size: u64,
}

#[derive(LintDiagnostic)]
#[diag(passes_repr_conflicting, code = E0566)]
pub(crate) struct ReprConflictingLint;

#[derive(Diagnostic)]
#[diag(passes_used_static)]
pub(crate) struct UsedStatic {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
    pub target: &'static str,
}

#[derive(Diagnostic)]
#[diag(passes_used_compiler_linker)]
pub(crate) struct UsedCompilerLinker {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_allow_internal_unstable)]
pub(crate) struct AllowInternalUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_debug_visualizer_placement)]
pub(crate) struct DebugVisualizerPlacement {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_debug_visualizer_invalid)]
#[note(passes_note_1)]
#[note(passes_note_2)]
#[note(passes_note_3)]
pub(crate) struct DebugVisualizerInvalid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_debug_visualizer_unreadable)]
pub(crate) struct DebugVisualizerUnreadable<'a> {
    #[primary_span]
    pub span: Span,
    pub file: &'a Path,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_allow_const_fn_unstable)]
pub(crate) struct RustcAllowConstFnUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_std_internal_symbol)]
pub(crate) struct RustcStdInternalSymbol {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_pub_transparent)]
pub(crate) struct RustcPubTransparent {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_force_inline)]
pub(crate) struct RustcForceInline {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_force_inline_coro)]
pub(crate) struct RustcForceInlineCoro {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_link_ordinal)]
pub(crate) struct LinkOrdinal {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_confusables)]
pub(crate) struct Confusables {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_coroutine_on_non_closure)]
pub(crate) struct CoroutineOnNonClosure {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_linkage)]
pub(crate) struct Linkage {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_stability_promotable)]
pub(crate) struct StabilityPromotable {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_deprecated)]
pub(crate) struct Deprecated;

#[derive(LintDiagnostic)]
#[diag(passes_macro_use)]
pub(crate) struct MacroUse {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
pub(crate) enum MacroExport {
    #[diag(passes_macro_export)]
    Normal,

    #[diag(passes_macro_export_on_decl_macro)]
    #[note]
    OnDeclMacro,

    #[diag(passes_invalid_macro_export_arguments)]
    InvalidArgument,

    #[diag(passes_invalid_macro_export_arguments_too_many_items)]
    TooManyItems,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedNote {
    #[note(passes_unused_empty_lints_note)]
    EmptyList { name: Symbol },
    #[note(passes_unused_no_lints_note)]
    NoLints { name: Symbol },
    #[note(passes_unused_default_method_body_const_note)]
    DefaultMethodBodyConst,
    #[note(passes_unused_linker_messages_note)]
    LinkerMessagesBinaryCrateOnly,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused)]
pub(crate) struct Unused {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub attr_span: Span,
    #[subdiagnostic]
    pub note: UnusedNote,
}

#[derive(Diagnostic)]
#[diag(passes_non_exported_macro_invalid_attrs, code = E0518)]
pub(crate) struct NonExportedMacroInvalidAttrs {
    #[primary_span]
    #[label]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_may_dangle)]
pub(crate) struct InvalidMayDangle {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_duplicate)]
pub(crate) struct UnusedDuplicate {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    #[warning]
    pub warning: bool,
}

#[derive(Diagnostic)]
#[diag(passes_unused_multiple)]
pub(crate) struct UnusedMultiple {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_lint_opt_ty)]
pub(crate) struct RustcLintOptTy {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_lint_opt_deny_field_access)]
pub(crate) struct RustcLintOptDenyFieldAccess {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_collapse_debuginfo)]
pub(crate) struct CollapseDebuginfo {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_deprecated_annotation_has_no_effect)]
pub(crate) struct DeprecatedAnnotationHasNoEffect {
    #[suggestion(applicability = "machine-applicable", code = "")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_unknown_external_lang_item, code = E0264)]
pub(crate) struct UnknownExternLangItem {
    #[primary_span]
    pub span: Span,
    pub lang_item: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_missing_panic_handler)]
pub(crate) struct MissingPanicHandler;

#[derive(Diagnostic)]
#[diag(passes_panic_unwind_without_std)]
#[help]
#[note]
pub(crate) struct PanicUnwindWithoutStd;

#[derive(Diagnostic)]
#[diag(passes_missing_lang_item)]
#[note]
#[help]
pub(crate) struct MissingLangItem {
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_lang_item_fn_with_track_caller)]
pub(crate) struct LangItemWithTrackCaller {
    #[primary_span]
    pub attr_span: Span,
    pub name: Symbol,
    #[label]
    pub sig_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_lang_item_fn_with_target_feature)]
pub(crate) struct LangItemWithTargetFeature {
    #[primary_span]
    pub attr_span: Span,
    pub name: Symbol,
    #[label]
    pub sig_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_lang_item_on_incorrect_target, code = E0718)]
pub(crate) struct LangItemOnIncorrectTarget {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
    pub expected_target: Target,
    pub actual_target: Target,
}

#[derive(Diagnostic)]
#[diag(passes_unknown_lang_item, code = E0522)]
pub(crate) struct UnknownLangItem {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
}

pub(crate) struct InvalidAttrAtCrateLevel {
    pub span: Span,
    pub sugg_span: Option<Span>,
    pub name: Symbol,
    pub item: Option<ItemFollowingInnerAttr>,
}

#[derive(Clone, Copy)]
pub(crate) struct ItemFollowingInnerAttr {
    pub span: Span,
    pub kind: &'static str,
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for InvalidAttrAtCrateLevel {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(dcx, level, fluent::passes_invalid_attr_at_crate_level);
        diag.span(self.span);
        diag.arg("name", self.name);
        // Only emit an error with a suggestion if we can create a string out
        // of the attribute span
        if let Some(span) = self.sugg_span {
            diag.span_suggestion_verbose(
                span,
                fluent::passes_suggestion,
                String::new(),
                Applicability::MachineApplicable,
            );
        }
        if let Some(item) = self.item {
            diag.arg("kind", item.kind);
            diag.span_label(item.span, fluent::passes_invalid_attr_at_crate_level_item);
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(passes_duplicate_diagnostic_item_in_crate)]
pub(crate) struct DuplicateDiagnosticItemInCrate {
    #[primary_span]
    pub duplicate_span: Option<Span>,
    #[note(passes_diagnostic_item_first_defined)]
    pub orig_span: Option<Span>,
    #[note]
    pub different_crates: bool,
    pub crate_name: Symbol,
    pub orig_crate_name: Symbol,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_layout_abi)]
pub(crate) struct LayoutAbi {
    #[primary_span]
    pub span: Span,
    pub abi: String,
}

#[derive(Diagnostic)]
#[diag(passes_layout_align)]
pub(crate) struct LayoutAlign {
    #[primary_span]
    pub span: Span,
    pub align: String,
}

#[derive(Diagnostic)]
#[diag(passes_layout_size)]
pub(crate) struct LayoutSize {
    #[primary_span]
    pub span: Span,
    pub size: String,
}

#[derive(Diagnostic)]
#[diag(passes_layout_homogeneous_aggregate)]
pub(crate) struct LayoutHomogeneousAggregate {
    #[primary_span]
    pub span: Span,
    pub homogeneous_aggregate: String,
}

#[derive(Diagnostic)]
#[diag(passes_layout_of)]
pub(crate) struct LayoutOf<'tcx> {
    #[primary_span]
    pub span: Span,
    pub normalized_ty: Ty<'tcx>,
    pub ty_layout: String,
}

#[derive(Diagnostic)]
#[diag(passes_layout_invalid_attribute)]
pub(crate) struct LayoutInvalidAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_abi_of)]
pub(crate) struct AbiOf {
    #[primary_span]
    pub span: Span,
    pub fn_name: Symbol,
    pub fn_abi: String,
}

#[derive(Diagnostic)]
#[diag(passes_abi_ne)]
pub(crate) struct AbiNe {
    #[primary_span]
    pub span: Span,
    pub left: String,
    pub right: String,
}

#[derive(Diagnostic)]
#[diag(passes_abi_invalid_attribute)]
pub(crate) struct AbiInvalidAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_unrecognized_argument)]
pub(crate) struct UnrecognizedArgument {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_feature_stable_twice, code = E0711)]
pub(crate) struct FeatureStableTwice {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub since: Symbol,
    pub prev_since: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_feature_previously_declared, code = E0711)]
pub(crate) struct FeaturePreviouslyDeclared<'a> {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub declared: &'a str,
    pub prev_declared: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_attr_only_in_functions)]
pub(crate) struct AttrOnlyInFunctions {
    #[primary_span]
    pub span: Span,
    pub attr: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_multiple_rustc_main, code = E0137)]
pub(crate) struct MultipleRustcMain {
    #[primary_span]
    pub span: Span,
    #[label(passes_first)]
    pub first: Span,
    #[label(passes_additional)]
    pub additional: Span,
}

#[derive(Diagnostic)]
#[diag(passes_extern_main)]
pub(crate) struct ExternMain {
    #[primary_span]
    pub span: Span,
}

pub(crate) struct NoMainErr {
    pub sp: Span,
    pub crate_name: Symbol,
    pub has_filename: bool,
    pub filename: PathBuf,
    pub file_empty: bool,
    pub non_main_fns: Vec<Span>,
    pub main_def_opt: Option<MainDefinition>,
    pub add_teach_note: bool,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for NoMainErr {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag = Diag::new(dcx, level, fluent::passes_no_main_function);
        diag.span(DUMMY_SP);
        diag.code(E0601);
        diag.arg("crate_name", self.crate_name);
        diag.arg("filename", self.filename);
        diag.arg("has_filename", self.has_filename);
        let note = if !self.non_main_fns.is_empty() {
            for &span in &self.non_main_fns {
                diag.span_note(span, fluent::passes_here_is_main);
            }
            diag.note(fluent::passes_one_or_more_possible_main);
            diag.help(fluent::passes_consider_moving_main);
            // There were some functions named `main` though. Try to give the user a hint.
            fluent::passes_main_must_be_defined_at_crate
        } else if self.has_filename {
            fluent::passes_consider_adding_main_to_file
        } else {
            fluent::passes_consider_adding_main_at_crate
        };
        if self.file_empty {
            diag.note(note);
        } else {
            diag.span(self.sp.shrink_to_hi());
            diag.span_label(self.sp.shrink_to_hi(), note);
        }

        if let Some(main_def) = self.main_def_opt
            && main_def.opt_fn_def_id().is_none()
        {
            // There is something at `crate::main`, but it is not a function definition.
            diag.span_label(main_def.span, fluent::passes_non_function_main);
        }

        if self.add_teach_note {
            diag.note(fluent::passes_teach_note);
        }
        diag
    }
}

pub(crate) struct DuplicateLangItem {
    pub local_span: Option<Span>,
    pub lang_item_name: Symbol,
    pub crate_name: Symbol,
    pub dependency_of: Option<Symbol>,
    pub is_local: bool,
    pub path: String,
    pub first_defined_span: Option<Span>,
    pub orig_crate_name: Option<Symbol>,
    pub orig_dependency_of: Option<Symbol>,
    pub orig_is_local: bool,
    pub orig_path: String,
    pub(crate) duplicate: Duplicate,
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for DuplicateLangItem {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            match self.duplicate {
                Duplicate::Plain => fluent::passes_duplicate_lang_item,
                Duplicate::Crate => fluent::passes_duplicate_lang_item_crate,
                Duplicate::CrateDepends => fluent::passes_duplicate_lang_item_crate_depends,
            },
        );
        diag.code(E0152);
        diag.arg("lang_item_name", self.lang_item_name);
        diag.arg("crate_name", self.crate_name);
        if let Some(dependency_of) = self.dependency_of {
            diag.arg("dependency_of", dependency_of);
        }
        diag.arg("path", self.path);
        if let Some(orig_crate_name) = self.orig_crate_name {
            diag.arg("orig_crate_name", orig_crate_name);
        }
        if let Some(orig_dependency_of) = self.orig_dependency_of {
            diag.arg("orig_dependency_of", orig_dependency_of);
        }
        diag.arg("orig_path", self.orig_path);
        if let Some(span) = self.local_span {
            diag.span(span);
        }
        if let Some(span) = self.first_defined_span {
            diag.span_note(span, fluent::passes_first_defined_span);
        } else {
            if self.orig_dependency_of.is_none() {
                diag.note(fluent::passes_first_defined_crate);
            } else {
                diag.note(fluent::passes_first_defined_crate_depends);
            }

            if self.orig_is_local {
                diag.note(fluent::passes_first_definition_local);
            } else {
                diag.note(fluent::passes_first_definition_path);
            }

            if self.is_local {
                diag.note(fluent::passes_second_definition_local);
            } else {
                diag.note(fluent::passes_second_definition_path);
            }
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(passes_incorrect_target, code = E0718)]
pub(crate) struct IncorrectTarget<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub generics_span: Span,
    pub name: &'a str, // cannot be symbol because it renders e.g. `r#fn` instead of `fn`
    pub kind: &'static str,
    pub num: usize,
    pub actual_num: usize,
    pub at_least: bool,
}

#[derive(Diagnostic)]
#[diag(passes_incorrect_crate_type)]
pub(crate) struct IncorrectCrateType {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_useless_assignment)]
pub(crate) struct UselessAssignment<'a> {
    pub is_field_assign: bool,
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(passes_only_has_effect_on)]
pub(crate) struct OnlyHasEffectOn {
    pub attr_name: String,
    pub target_name: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_inline_ignored_for_exported)]
#[help]
pub(crate) struct InlineIgnoredForExported {}

#[derive(Diagnostic)]
#[diag(passes_object_lifetime_err)]
pub(crate) struct ObjectLifetimeErr {
    #[primary_span]
    pub span: Span,
    pub repr: String,
}

#[derive(Diagnostic)]
pub(crate) enum AttrApplication {
    #[diag(passes_attr_application_enum, code = E0517)]
    Enum {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes_attr_application_struct, code = E0517)]
    Struct {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes_attr_application_struct_union, code = E0517)]
    StructUnion {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes_attr_application_struct_enum_union, code = E0517)]
    StructEnumUnion {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(passes_transparent_incompatible, code = E0692)]
pub(crate) struct TransparentIncompatible {
    #[primary_span]
    pub hint_spans: Vec<Span>,
    pub target: String,
}

#[derive(Diagnostic)]
#[diag(passes_deprecated_attribute, code = E0549)]
pub(crate) struct DeprecatedAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_useless_stability)]
pub(crate) struct UselessStability {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes_cannot_stabilize_deprecated)]
pub(crate) struct CannotStabilizeDeprecated {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes_unstable_attr_for_already_stable_feature)]
pub(crate) struct UnstableAttrForAlreadyStableFeature {
    #[primary_span]
    #[label]
    #[help]
    pub span: Span,
    #[label(passes_item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes_missing_stability_attr)]
pub(crate) struct MissingStabilityAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_missing_const_stab_attr)]
pub(crate) struct MissingConstStabAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_trait_impl_const_stable)]
#[note]
pub(crate) struct TraitImplConstStable {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_trait_impl_const_stability_mismatch)]
pub(crate) struct TraitImplConstStabilityMismatch {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub impl_stability: ImplConstStability,
    #[subdiagnostic]
    pub trait_stability: TraitConstStability,
}

#[derive(Subdiagnostic)]
pub(crate) enum TraitConstStability {
    #[note(passes_trait_impl_const_stability_mismatch_trait_stable)]
    Stable {
        #[primary_span]
        span: Span,
    },
    #[note(passes_trait_impl_const_stability_mismatch_trait_unstable)]
    Unstable {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum ImplConstStability {
    #[note(passes_trait_impl_const_stability_mismatch_impl_stable)]
    Stable {
        #[primary_span]
        span: Span,
    },
    #[note(passes_trait_impl_const_stability_mismatch_impl_unstable)]
    Unstable {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(passes_unknown_feature, code = E0635)]
pub(crate) struct UnknownFeature {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_unknown_feature_alias, code = E0635)]
pub(crate) struct RenamedFeature {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub alias: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_implied_feature_not_exist)]
pub(crate) struct ImpliedFeatureNotExist {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub implied_by: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_duplicate_feature_err, code = E0636)]
pub(crate) struct DuplicateFeatureErr {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_missing_const_err)]
pub(crate) struct MissingConstErr {
    #[primary_span]
    #[help]
    pub fn_sig_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_const_stable_not_stable)]
pub(crate) struct ConstStableNotStable {
    #[primary_span]
    pub fn_sig_span: Span,
    #[label]
    pub const_span: Span,
}

#[derive(LintDiagnostic)]
pub(crate) enum MultipleDeadCodes<'tcx> {
    #[diag(passes_dead_codes)]
    DeadCodes {
        multiple: bool,
        num: usize,
        descr: &'tcx str,
        participle: &'tcx str,
        name_list: DiagSymbolList,
        #[subdiagnostic]
        // only on DeadCodes since it's never a problem for tuple struct fields
        enum_variants_with_same_name: Vec<EnumVariantSameName<'tcx>>,
        #[subdiagnostic]
        parent_info: Option<ParentInfo<'tcx>>,
        #[subdiagnostic]
        ignored_derived_impls: Option<IgnoredDerivedImpls>,
    },
    #[diag(passes_dead_codes)]
    UnusedTupleStructFields {
        multiple: bool,
        num: usize,
        descr: &'tcx str,
        participle: &'tcx str,
        name_list: DiagSymbolList,
        #[subdiagnostic]
        change_fields_suggestion: ChangeFields,
        #[subdiagnostic]
        parent_info: Option<ParentInfo<'tcx>>,
        #[subdiagnostic]
        ignored_derived_impls: Option<IgnoredDerivedImpls>,
    },
}

#[derive(Subdiagnostic)]
#[note(passes_enum_variant_same_name)]
pub(crate) struct EnumVariantSameName<'tcx> {
    #[primary_span]
    pub variant_span: Span,
    pub dead_name: Symbol,
    pub dead_descr: &'tcx str,
}

#[derive(Subdiagnostic)]
#[label(passes_parent_info)]
pub(crate) struct ParentInfo<'tcx> {
    pub num: usize,
    pub descr: &'tcx str,
    pub parent_descr: &'tcx str,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[note(passes_ignored_derived_impls)]
pub(crate) struct IgnoredDerivedImpls {
    pub name: Symbol,
    pub trait_list: DiagSymbolList,
    pub trait_list_len: usize,
}

#[derive(Subdiagnostic)]
pub(crate) enum ChangeFields {
    #[multipart_suggestion(
        passes_change_fields_to_be_of_unit_type,
        applicability = "has-placeholders"
    )]
    ChangeToUnitTypeOrRemove {
        num: usize,
        #[suggestion_part(code = "()")]
        spans: Vec<Span>,
    },
    #[help(passes_remove_fields)]
    Remove { num: usize },
}

#[derive(Diagnostic)]
#[diag(passes_proc_macro_bad_sig)]
pub(crate) struct ProcMacroBadSig {
    #[primary_span]
    pub span: Span,
    pub kind: ProcMacroKind,
}

#[derive(LintDiagnostic)]
#[diag(passes_unreachable_due_to_uninhabited)]
pub(crate) struct UnreachableDueToUninhabited<'desc, 'tcx> {
    pub descr: &'desc str,
    #[label]
    pub expr: Span,
    #[label(passes_label_orig)]
    #[note]
    pub orig: Span,
    pub ty: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_var_maybe_capture_ref)]
#[help]
pub(crate) struct UnusedVarMaybeCaptureRef {
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_capture_maybe_capture_ref)]
#[help]
pub(crate) struct UnusedCaptureMaybeCaptureRef {
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_var_remove_field)]
pub(crate) struct UnusedVarRemoveField {
    pub name: String,
    #[subdiagnostic]
    pub sugg: UnusedVarRemoveFieldSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    passes_unused_var_remove_field_suggestion,
    applicability = "machine-applicable"
)]
pub(crate) struct UnusedVarRemoveFieldSugg {
    #[suggestion_part(code = "")]
    pub spans: Vec<Span>,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_var_assigned_only)]
#[note]
pub(crate) struct UnusedVarAssignedOnly {
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_unnecessary_stable_feature)]
pub(crate) struct UnnecessaryStableFeature {
    pub feature: Symbol,
    pub since: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(passes_unnecessary_partial_stable_feature)]
pub(crate) struct UnnecessaryPartialStableFeature {
    #[suggestion(code = "{implies}", applicability = "maybe-incorrect")]
    pub span: Span,
    #[suggestion(passes_suggestion_remove, code = "", applicability = "maybe-incorrect")]
    pub line: Span,
    pub feature: Symbol,
    pub since: Symbol,
    pub implies: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(passes_ineffective_unstable_impl)]
#[note]
pub(crate) struct IneffectiveUnstableImpl;

#[derive(LintDiagnostic)]
#[diag(passes_unused_assign)]
pub(crate) struct UnusedAssign {
    pub name: String,
    #[subdiagnostic]
    pub suggestion: Option<UnusedAssignSuggestion>,
    #[help]
    pub help: bool,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(passes_unused_assign_suggestion, applicability = "maybe-incorrect")]
pub(crate) struct UnusedAssignSuggestion {
    pub pre: &'static str,
    #[suggestion_part(code = "{pre}mut ")]
    pub ty_span: Option<Span>,
    #[suggestion_part(code = "")]
    pub ty_ref_span: Span,
    #[suggestion_part(code = "*")]
    pub ident_span: Span,
    #[suggestion_part(code = "")]
    pub expr_ref_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_assign_passed)]
#[help]
pub(crate) struct UnusedAssignPassed {
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_variable_try_prefix)]
pub(crate) struct UnusedVariableTryPrefix {
    #[label]
    pub label: Option<Span>,
    #[subdiagnostic]
    pub string_interp: Vec<UnusedVariableStringInterp>,
    #[subdiagnostic]
    pub sugg: UnusedVariableSugg,
    pub name: String,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedVariableSugg {
    #[multipart_suggestion(passes_suggestion, applicability = "maybe-incorrect")]
    TryPrefixSugg {
        #[suggestion_part(code = "_{name}")]
        spans: Vec<Span>,
        name: String,
    },
    #[help(passes_unused_variable_args_in_macro)]
    NoSugg {
        #[primary_span]
        span: Span,
        name: String,
    },
}

pub(crate) struct UnusedVariableStringInterp {
    pub lit: Span,
    pub lo: Span,
    pub hi: Span,
}

impl Subdiagnostic for UnusedVariableStringInterp {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.span_label(self.lit, crate::fluent_generated::passes_maybe_string_interpolation);
        diag.multipart_suggestion(
            crate::fluent_generated::passes_string_interpolation_only_works,
            vec![(self.lo, String::from("format!(")), (self.hi, String::from(")"))],
            Applicability::MachineApplicable,
        );
    }
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_variable_try_ignore)]
pub(crate) struct UnusedVarTryIgnore {
    pub name: String,
    #[subdiagnostic]
    pub sugg: UnusedVarTryIgnoreSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(passes_suggestion, applicability = "maybe-incorrect")]
pub(crate) struct UnusedVarTryIgnoreSugg {
    #[suggestion_part(code = "{name}: _")]
    pub shorthands: Vec<Span>,
    #[suggestion_part(code = "_")]
    pub non_shorthands: Vec<Span>,
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_attr_crate_level)]
#[note]
pub(crate) struct AttrCrateLevelOnly {
    #[subdiagnostic]
    pub sugg: Option<AttrCrateLevelOnlySugg>,
}

#[derive(Subdiagnostic)]
#[suggestion(passes_suggestion, applicability = "maybe-incorrect", code = "!", style = "verbose")]
pub(crate) struct AttrCrateLevelOnlySugg {
    #[primary_span]
    pub attr: Span,
}

#[derive(Diagnostic)]
#[diag(passes_no_sanitize)]
pub(crate) struct NoSanitize<'a> {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
    pub accepted_kind: &'a str,
    pub attr_str: &'a str,
}

// FIXME(jdonszelmann): move back to rustc_attr
#[derive(Diagnostic)]
#[diag(passes_rustc_const_stable_indirect_pairing)]
pub(crate) struct RustcConstStableIndirectPairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_unsupported_attributes_in_where)]
#[help]
pub(crate) struct UnsupportedAttributesInWhere {
    #[primary_span]
    pub span: MultiSpan,
}

#[derive(Diagnostic)]
pub(crate) enum UnexportableItem<'a> {
    #[diag(passes_unexportable_item)]
    Item {
        #[primary_span]
        span: Span,
        descr: &'a str,
    },

    #[diag(passes_unexportable_generic_fn)]
    GenericFn(#[primary_span] Span),

    #[diag(passes_unexportable_fn_abi)]
    FnAbi(#[primary_span] Span),

    #[diag(passes_unexportable_type_repr)]
    TypeRepr(#[primary_span] Span),

    #[diag(passes_unexportable_type_in_interface)]
    TypeInInterface {
        #[primary_span]
        span: Span,
        desc: &'a str,
        ty: &'a str,
        #[label]
        ty_span: Span,
    },

    #[diag(passes_unexportable_priv_item)]
    PrivItem {
        #[primary_span]
        span: Span,
        #[note]
        vis_note: Span,
        vis_descr: &'a str,
    },

    #[diag(passes_unexportable_adt_with_private_fields)]
    AdtWithPrivFields {
        #[primary_span]
        span: Span,
        #[note]
        vis_note: Span,
        field_name: &'a str,
    },
}

#[derive(Diagnostic)]
#[diag(passes_repr_align_should_be_align)]
pub(crate) struct ReprAlignShouldBeAlign {
    #[primary_span]
    #[help]
    pub span: Span,
    pub item: &'static str,
}

#[derive(Diagnostic)]
#[diag(passes_align_should_be_repr_align)]
pub(crate) struct AlignShouldBeReprAlign {
    #[primary_span]
    #[suggestion(
        style = "verbose",
        applicability = "machine-applicable",
        code = "#[repr(align({align_bytes}))]"
    )]
    pub span: Span,
    pub item: &'static str,
    pub align_bytes: u64,
}
