use std::{
    io::Error,
    path::{Path, PathBuf},
};

use crate::fluent_generated as fluent;
use rustc_ast::Label;
use rustc_errors::{
    error_code, Applicability, DiagnosticSymbolList, ErrorGuaranteed, IntoDiagnostic, MultiSpan,
};
use rustc_hir::{self as hir, ExprKind, Target};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{MainDefinition, Ty};
use rustc_span::{Span, Symbol, DUMMY_SP};

use crate::check_attr::ProcMacroKind;
use crate::lang_items::Duplicate;

#[derive(Diagnostic)]
#[diag(passes_incorrect_do_not_recommend_location)]
pub struct IncorrectDoNotRecommendLocation {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_outer_crate_level_attr)]
pub struct OuterCrateLevelAttr;

#[derive(LintDiagnostic)]
#[diag(passes_inner_crate_level_attr)]
pub struct InnerCrateLevelAttr;

#[derive(LintDiagnostic)]
#[diag(passes_ignored_attr_with_macro)]
pub struct IgnoredAttrWithMacro<'a> {
    pub sym: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes_ignored_attr)]
pub struct IgnoredAttr<'a> {
    pub sym: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes_inline_ignored_function_prototype)]
pub struct IgnoredInlineAttrFnProto;

#[derive(LintDiagnostic)]
#[diag(passes_inline_ignored_constants)]
#[warning]
#[note]
pub struct IgnoredInlineAttrConstants;

#[derive(Diagnostic)]
#[diag(passes_inline_not_fn_or_closure, code = "E0518")]
pub struct InlineNotFnOrClosure {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_no_coverage_ignored_function_prototype)]
pub struct IgnoredNoCoverageFnProto;

#[derive(LintDiagnostic)]
#[diag(passes_no_coverage_propagate)]
pub struct IgnoredNoCoveragePropagate;

#[derive(LintDiagnostic)]
#[diag(passes_no_coverage_fn_defn)]
pub struct IgnoredNoCoverageFnDefn;

#[derive(Diagnostic)]
#[diag(passes_no_coverage_not_coverable, code = "E0788")]
pub struct IgnoredNoCoverageNotCoverable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_fn)]
pub struct AttrShouldBeAppliedToFn {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
    pub on_crate: bool,
}

#[derive(Diagnostic)]
#[diag(passes_naked_tracked_caller, code = "E0736")]
pub struct NakedTrackedCaller {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_fn, code = "E0739")]
pub struct TrackedCallerWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
    pub on_crate: bool,
}

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_struct_enum, code = "E0701")]
pub struct NonExhaustiveWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_trait)]
pub struct AttrShouldBeAppliedToTrait {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_target_feature_on_statement)]
pub struct TargetFeatureOnStatement;

#[derive(Diagnostic)]
#[diag(passes_should_be_applied_to_static)]
pub struct AttrShouldBeAppliedToStatic {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_expect_str)]
pub struct DocExpectStr<'a> {
    #[primary_span]
    pub attr_span: Span,
    pub attr_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_empty)]
pub struct DocAliasEmpty<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_bad_char)]
pub struct DocAliasBadChar<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub char_: char,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_start_end)]
pub struct DocAliasStartEnd<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_bad_location)]
pub struct DocAliasBadLocation<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub location: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_not_an_alias)]
pub struct DocAliasNotAnAlias<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_alias_duplicated)]
pub struct DocAliasDuplicated {
    #[label]
    pub first_defn: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_not_string_literal)]
pub struct DocAliasNotStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_alias_malformed)]
pub struct DocAliasMalformed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_keyword_empty_mod)]
pub struct DocKeywordEmptyMod {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_keyword_not_mod)]
pub struct DocKeywordNotMod {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_keyword_invalid_ident)]
pub struct DocKeywordInvalidIdent {
    #[primary_span]
    pub span: Span,
    pub doc_keyword: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_doc_fake_variadic_not_valid)]
pub struct DocFakeVariadicNotValid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_keyword_only_impl)]
pub struct DocKeywordOnlyImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_doc_inline_conflict)]
#[help]
pub struct DocKeywordConflict {
    #[primary_span]
    pub spans: MultiSpan,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_inline_only_use)]
#[note]
pub struct DocInlineOnlyUse {
    #[label]
    pub attr_span: Span,
    #[label(passes_not_a_use_item_label)]
    pub item_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_doc_attr_not_crate_level)]
pub struct DocAttrNotCrateLevel<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_name: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown)]
pub struct DocTestUnknown {
    pub path: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_takes_list)]
pub struct DocTestTakesList;

#[derive(LintDiagnostic)]
#[diag(passes_doc_cfg_hide_takes_list)]
pub struct DocCfgHideTakesList;

#[derive(LintDiagnostic)]
#[diag(passes_doc_primitive)]
pub struct DocPrimitive;

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown_any)]
pub struct DocTestUnknownAny {
    pub path: String,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown_spotlight)]
#[note]
#[note(passes_no_op_note)]
pub struct DocTestUnknownSpotlight {
    pub path: String,
    #[suggestion(style = "short", applicability = "machine-applicable", code = "notable_trait")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_test_unknown_include)]
pub struct DocTestUnknownInclude {
    pub path: String,
    pub value: String,
    pub inner: &'static str,
    #[suggestion(code = "#{inner}[doc = include_str!(\"{value}\")]")]
    pub sugg: (Span, Applicability),
}

#[derive(LintDiagnostic)]
#[diag(passes_doc_invalid)]
pub struct DocInvalid;

#[derive(Diagnostic)]
#[diag(passes_pass_by_value)]
pub struct PassByValue {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_allow_incoherent_impl)]
pub struct AllowIncoherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_has_incoherent_inherent_impl)]
pub struct HasIncoherentInherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_both_ffi_const_and_pure, code = "E0757")]
pub struct BothFfiConstAndPure {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_ffi_pure_invalid_target, code = "E0755")]
pub struct FfiPureInvalidTarget {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_ffi_const_invalid_target, code = "E0756")]
pub struct FfiConstInvalidTarget {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_ffi_returns_twice_invalid_target, code = "E0724")]
pub struct FfiReturnsTwiceInvalidTarget {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_must_use_async)]
pub struct MustUseAsync {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_must_use_no_effect)]
pub struct MustUseNoEffect {
    pub article: &'static str,
    pub target: rustc_hir::Target,
}

#[derive(Diagnostic)]
#[diag(passes_must_not_suspend)]
pub struct MustNotSuspend {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_cold)]
#[warning]
pub struct Cold {
    #[label]
    pub span: Span,
    pub on_crate: bool,
}

#[derive(LintDiagnostic)]
#[diag(passes_link)]
#[warning]
pub struct Link {
    #[label]
    pub span: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(passes_link_name)]
#[warning]
pub struct LinkName<'a> {
    #[help]
    pub attr_span: Option<Span>,
    #[label]
    pub span: Span,
    pub value: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_no_link)]
pub struct NoLink {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_export_name)]
pub struct ExportName {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_layout_scalar_valid_range_not_struct)]
pub struct RustcLayoutScalarValidRangeNotStruct {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_layout_scalar_valid_range_arg)]
pub struct RustcLayoutScalarValidRangeArg {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_legacy_const_generics_only)]
pub struct RustcLegacyConstGenericsOnly {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub param_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_legacy_const_generics_index)]
pub struct RustcLegacyConstGenericsIndex {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub generics_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_legacy_const_generics_index_exceed)]
pub struct RustcLegacyConstGenericsIndexExceed {
    #[primary_span]
    #[label]
    pub span: Span,
    pub arg_count: usize,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_legacy_const_generics_index_negative)]
pub struct RustcLegacyConstGenericsIndexNegative {
    #[primary_span]
    pub invalid_args: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_dirty_clean)]
pub struct RustcDirtyClean {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_link_section)]
#[warning]
pub struct LinkSection {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_no_mangle_foreign)]
#[warning]
#[note]
pub struct NoMangleForeign {
    #[label]
    pub span: Span,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub attr_span: Span,
    pub foreign_item_kind: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(passes_no_mangle)]
#[warning]
pub struct NoMangle {
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_repr_ident, code = "E0565")]
pub struct ReprIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_repr_conflicting, code = "E0566")]
pub struct ReprConflicting;

#[derive(Diagnostic)]
#[diag(passes_used_static)]
pub struct UsedStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_used_compiler_linker)]
pub struct UsedCompilerLinker {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_allow_internal_unstable)]
pub struct AllowInternalUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_debug_visualizer_placement)]
pub struct DebugVisualizerPlacement {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_debug_visualizer_invalid)]
#[note(passes_note_1)]
#[note(passes_note_2)]
#[note(passes_note_3)]
pub struct DebugVisualizerInvalid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_debug_visualizer_unreadable)]
pub struct DebugVisualizerUnreadable<'a> {
    #[primary_span]
    pub span: Span,
    pub file: &'a Path,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_allow_const_fn_unstable)]
pub struct RustcAllowConstFnUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_std_internal_symbol)]
pub struct RustcStdInternalSymbol {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_const_trait)]
pub struct ConstTrait {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_link_ordinal)]
pub struct LinkOrdinal {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_stability_promotable)]
pub struct StabilityPromotable {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_deprecated)]
pub struct Deprecated;

#[derive(LintDiagnostic)]
#[diag(passes_macro_use)]
pub struct MacroUse {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
pub enum MacroExport {
    #[diag(passes_macro_export)]
    Normal,

    #[diag(passes_invalid_macro_export_arguments)]
    UnknownItem { name: Symbol },

    #[diag(passes_invalid_macro_export_arguments_too_many_items)]
    TooManyItems,
}

#[derive(LintDiagnostic)]
#[diag(passes_plugin_registrar)]
pub struct PluginRegistrar;

#[derive(Subdiagnostic)]
pub enum UnusedNote {
    #[note(passes_unused_empty_lints_note)]
    EmptyList { name: Symbol },
    #[note(passes_unused_no_lints_note)]
    NoLints { name: Symbol },
    #[note(passes_unused_default_method_body_const_note)]
    DefaultMethodBodyConst,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused)]
pub struct Unused {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub attr_span: Span,
    #[subdiagnostic]
    pub note: UnusedNote,
}

#[derive(Diagnostic)]
#[diag(passes_non_exported_macro_invalid_attrs, code = "E0518")]
pub struct NonExportedMacroInvalidAttrs {
    #[primary_span]
    #[label]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_unused_duplicate)]
pub struct UnusedDuplicate {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    #[warning]
    pub warning: Option<()>,
}

#[derive(Diagnostic)]
#[diag(passes_unused_multiple)]
pub struct UnusedMultiple {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_lint_opt_ty)]
pub struct RustcLintOptTy {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_rustc_lint_opt_deny_field_access)]
pub struct RustcLintOptDenyFieldAccess {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_collapse_debuginfo)]
pub struct CollapseDebuginfo {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_deprecated_annotation_has_no_effect)]
pub struct DeprecatedAnnotationHasNoEffect {
    #[suggestion(applicability = "machine-applicable", code = "")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_unknown_external_lang_item, code = "E0264")]
pub struct UnknownExternLangItem {
    #[primary_span]
    pub span: Span,
    pub lang_item: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_missing_panic_handler)]
pub struct MissingPanicHandler;

#[derive(Diagnostic)]
#[diag(passes_missing_lang_item)]
#[note]
#[help]
pub struct MissingLangItem {
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_lang_item_on_incorrect_target, code = "E0718")]
pub struct LangItemOnIncorrectTarget {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
    pub expected_target: Target,
    pub actual_target: Target,
}

#[derive(Diagnostic)]
#[diag(passes_unknown_lang_item, code = "E0522")]
pub struct UnknownLangItem {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
}

pub struct InvalidAttrAtCrateLevel {
    pub span: Span,
    pub snippet: Option<String>,
    pub name: Symbol,
}

impl IntoDiagnostic<'_> for InvalidAttrAtCrateLevel {
    #[track_caller]
    fn into_diagnostic(
        self,
        handler: &'_ rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::passes_invalid_attr_at_crate_level);
        diag.set_span(self.span);
        diag.set_arg("name", self.name);
        // Only emit an error with a suggestion if we can create a string out
        // of the attribute span
        if let Some(src) = self.snippet {
            let replacement = src.replace("#!", "#");
            diag.span_suggestion_verbose(
                self.span,
                fluent::passes_suggestion,
                replacement,
                rustc_errors::Applicability::MachineApplicable,
            );
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(passes_duplicate_diagnostic_item_in_crate)]
pub struct DuplicateDiagnosticItemInCrate {
    #[primary_span]
    pub duplicate_span: Option<Span>,
    #[note(passes_diagnostic_item_first_defined)]
    pub orig_span: Option<Span>,
    #[note]
    pub different_crates: Option<()>,
    pub crate_name: Symbol,
    pub orig_crate_name: Symbol,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_abi)]
pub struct Abi {
    #[primary_span]
    pub span: Span,
    pub abi: String,
}

#[derive(Diagnostic)]
#[diag(passes_align)]
pub struct Align {
    #[primary_span]
    pub span: Span,
    pub align: String,
}

#[derive(Diagnostic)]
#[diag(passes_size)]
pub struct Size {
    #[primary_span]
    pub span: Span,
    pub size: String,
}

#[derive(Diagnostic)]
#[diag(passes_homogeneous_aggregate)]
pub struct HomogeneousAggregate {
    #[primary_span]
    pub span: Span,
    pub homogeneous_aggregate: String,
}

#[derive(Diagnostic)]
#[diag(passes_layout_of)]
pub struct LayoutOf {
    #[primary_span]
    pub span: Span,
    pub normalized_ty: String,
    pub ty_layout: String,
}

#[derive(Diagnostic)]
#[diag(passes_unrecognized_field)]
pub struct UnrecognizedField {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_feature_stable_twice, code = "E0711")]
pub struct FeatureStableTwice {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub since: Symbol,
    pub prev_since: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_feature_previously_declared, code = "E0711")]
pub struct FeaturePreviouslyDeclared<'a, 'b> {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub declared: &'a str,
    pub prev_declared: &'b str,
}

#[derive(Diagnostic)]
#[diag(passes_expr_not_allowed_in_context, code = "E0744")]
pub struct ExprNotAllowedInContext<'a> {
    #[primary_span]
    pub span: Span,
    pub expr: String,
    pub context: &'a str,
}

pub struct BreakNonLoop<'a> {
    pub span: Span,
    pub head: Option<Span>,
    pub kind: &'a str,
    pub suggestion: String,
    pub loop_label: Option<Label>,
    pub break_label: Option<Label>,
    pub break_expr_kind: &'a ExprKind<'a>,
    pub break_expr_span: Span,
}

impl<'a> IntoDiagnostic<'_> for BreakNonLoop<'a> {
    #[track_caller]
    fn into_diagnostic(
        self,
        handler: &rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_span_err_with_code(
            self.span,
            fluent::passes_break_non_loop,
            error_code!(E0571),
        );
        diag.set_arg("kind", self.kind);
        diag.span_label(self.span, fluent::passes_label);
        if let Some(head) = self.head {
            diag.span_label(head, fluent::passes_label2);
        }
        diag.span_suggestion(
            self.span,
            fluent::passes_suggestion,
            self.suggestion,
            Applicability::MaybeIncorrect,
        );
        if let (Some(label), None) = (self.loop_label, self.break_label) {
            match self.break_expr_kind {
                ExprKind::Path(hir::QPath::Resolved(
                    None,
                    hir::Path { segments: [segment], res: hir::def::Res::Err, .. },
                )) if label.ident.to_string() == format!("'{}", segment.ident) => {
                    // This error is redundant, we will have already emitted a
                    // suggestion to use the label when `segment` wasn't found
                    // (hence the `Res::Err` check).
                    diag.delay_as_bug();
                }
                _ => {
                    diag.span_suggestion(
                        self.break_expr_span,
                        fluent::passes_break_expr_suggestion,
                        label.ident,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(passes_continue_labeled_block, code = "E0696")]
pub struct ContinueLabeledBlock {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_block_label)]
    pub block_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_break_inside_closure, code = "E0267")]
pub struct BreakInsideClosure<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_closure_label)]
    pub closure_span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_break_inside_async_block, code = "E0267")]
pub struct BreakInsideAsyncBlock<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_async_block_label)]
    pub closure_span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_outside_loop, code = "E0268")]
pub struct OutsideLoop<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: &'a str,
    pub is_break: bool,
}

#[derive(Diagnostic)]
#[diag(passes_unlabeled_in_labeled_block, code = "E0695")]
pub struct UnlabeledInLabeledBlock<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub cf_type: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_unlabeled_cf_in_while_condition, code = "E0590")]
pub struct UnlabeledCfInWhileCondition<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub cf_type: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_cannot_inline_naked_function)]
pub struct CannotInlineNakedFunction {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes_undefined_naked_function_abi)]
pub struct UndefinedNakedFunctionAbi;

#[derive(Diagnostic)]
#[diag(passes_no_patterns)]
pub struct NoPatterns {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_params_not_allowed)]
#[help]
pub struct ParamsNotAllowed {
    #[primary_span]
    pub span: Span,
}

pub struct NakedFunctionsAsmBlock {
    pub span: Span,
    pub multiple_asms: Vec<Span>,
    pub non_asms: Vec<Span>,
}

impl IntoDiagnostic<'_> for NakedFunctionsAsmBlock {
    #[track_caller]
    fn into_diagnostic(
        self,
        handler: &rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_span_err_with_code(
            self.span,
            fluent::passes_naked_functions_asm_block,
            error_code!(E0787),
        );
        for span in self.multiple_asms.iter() {
            diag.span_label(*span, fluent::passes_label_multiple_asm);
        }
        for span in self.non_asms.iter() {
            diag.span_label(*span, fluent::passes_label_non_asm);
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(passes_naked_functions_operands, code = "E0787")]
pub struct NakedFunctionsOperands {
    #[primary_span]
    pub unsupported_operands: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_naked_functions_asm_options, code = "E0787")]
pub struct NakedFunctionsAsmOptions {
    #[primary_span]
    pub span: Span,
    pub unsupported_options: String,
}

#[derive(Diagnostic)]
#[diag(passes_naked_functions_must_use_noreturn, code = "E0787")]
pub struct NakedFunctionsMustUseNoreturn {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = ", options(noreturn)", applicability = "machine-applicable")]
    pub last_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_attr_only_on_main)]
pub struct AttrOnlyOnMain {
    #[primary_span]
    pub span: Span,
    pub attr: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_attr_only_on_root_main)]
pub struct AttrOnlyOnRootMain {
    #[primary_span]
    pub span: Span,
    pub attr: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_attr_only_in_functions)]
pub struct AttrOnlyInFunctions {
    #[primary_span]
    pub span: Span,
    pub attr: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_multiple_rustc_main, code = "E0137")]
pub struct MultipleRustcMain {
    #[primary_span]
    pub span: Span,
    #[label(passes_first)]
    pub first: Span,
    #[label(passes_additional)]
    pub additional: Span,
}

#[derive(Diagnostic)]
#[diag(passes_multiple_start_functions, code = "E0138")]
pub struct MultipleStartFunctions {
    #[primary_span]
    pub span: Span,
    #[label]
    pub labeled: Span,
    #[label(passes_previous)]
    pub previous: Span,
}

#[derive(Diagnostic)]
#[diag(passes_extern_main)]
pub struct ExternMain {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_unix_sigpipe_values)]
pub struct UnixSigpipeValues {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_no_main_function, code = "E0601")]
pub struct NoMainFunction {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
}

pub struct NoMainErr {
    pub sp: Span,
    pub crate_name: Symbol,
    pub has_filename: bool,
    pub filename: PathBuf,
    pub file_empty: bool,
    pub non_main_fns: Vec<Span>,
    pub main_def_opt: Option<MainDefinition>,
    pub add_teach_note: bool,
}

impl<'a> IntoDiagnostic<'a> for NoMainErr {
    #[track_caller]
    fn into_diagnostic(
        self,
        handler: &'a rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'a, ErrorGuaranteed> {
        let mut diag = handler.struct_span_err_with_code(
            DUMMY_SP,
            fluent::passes_no_main_function,
            error_code!(E0601),
        );
        diag.set_arg("crate_name", self.crate_name);
        diag.set_arg("filename", self.filename);
        diag.set_arg("has_filename", self.has_filename);
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
            diag.set_span(self.sp.shrink_to_hi());
            diag.span_label(self.sp.shrink_to_hi(), note);
        }

        if let Some(main_def) = self.main_def_opt && main_def.opt_fn_def_id().is_none(){
            // There is something at `crate::main`, but it is not a function definition.
            diag.span_label(main_def.span, fluent::passes_non_function_main);
        }

        if self.add_teach_note {
            diag.note(fluent::passes_teach_note);
        }
        diag
    }
}

pub struct DuplicateLangItem {
    pub local_span: Option<Span>,
    pub lang_item_name: Symbol,
    pub crate_name: Symbol,
    pub dependency_of: Symbol,
    pub is_local: bool,
    pub path: String,
    pub first_defined_span: Option<Span>,
    pub orig_crate_name: Symbol,
    pub orig_dependency_of: Symbol,
    pub orig_is_local: bool,
    pub orig_path: String,
    pub(crate) duplicate: Duplicate,
}

impl IntoDiagnostic<'_> for DuplicateLangItem {
    #[track_caller]
    fn into_diagnostic(
        self,
        handler: &rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err_with_code(
            match self.duplicate {
                Duplicate::Plain => fluent::passes_duplicate_lang_item,
                Duplicate::Crate => fluent::passes_duplicate_lang_item_crate,
                Duplicate::CrateDepends => fluent::passes_duplicate_lang_item_crate_depends,
            },
            error_code!(E0152),
        );
        diag.set_arg("lang_item_name", self.lang_item_name);
        diag.set_arg("crate_name", self.crate_name);
        diag.set_arg("dependency_of", self.dependency_of);
        diag.set_arg("path", self.path);
        diag.set_arg("orig_crate_name", self.orig_crate_name);
        diag.set_arg("orig_dependency_of", self.orig_dependency_of);
        diag.set_arg("orig_path", self.orig_path);
        if let Some(span) = self.local_span {
            diag.set_span(span);
        }
        if let Some(span) = self.first_defined_span {
            diag.span_note(span, fluent::passes_first_defined_span);
        } else {
            if self.orig_dependency_of.is_empty() {
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
#[diag(passes_incorrect_target, code = "E0718")]
pub struct IncorrectTarget<'a> {
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

#[derive(LintDiagnostic)]
#[diag(passes_useless_assignment)]
pub struct UselessAssignment<'a> {
    pub is_field_assign: bool,
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(passes_only_has_effect_on)]
pub struct OnlyHasEffectOn {
    pub attr_name: Symbol,
    pub target_name: String,
}

#[derive(Diagnostic)]
#[diag(passes_object_lifetime_err)]
pub struct ObjectLifetimeErr {
    #[primary_span]
    pub span: Span,
    pub repr: String,
}

#[derive(Diagnostic)]
#[diag(passes_unrecognized_repr_hint, code = "E0552")]
#[help]
pub struct UnrecognizedReprHint {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
pub enum AttrApplication {
    #[diag(passes_attr_application_enum, code = "E0517")]
    Enum {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes_attr_application_struct, code = "E0517")]
    Struct {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes_attr_application_struct_union, code = "E0517")]
    StructUnion {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes_attr_application_struct_enum_union, code = "E0517")]
    StructEnumUnion {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes_attr_application_struct_enum_function_union, code = "E0517")]
    StructEnumFunctionUnion {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(passes_transparent_incompatible, code = "E0692")]
pub struct TransparentIncompatible {
    #[primary_span]
    pub hint_spans: Vec<Span>,
    pub target: String,
}

#[derive(Diagnostic)]
#[diag(passes_deprecated_attribute, code = "E0549")]
pub struct DeprecatedAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_useless_stability)]
pub struct UselessStability {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes_invalid_stability)]
pub struct InvalidStability {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes_cannot_stabilize_deprecated)]
pub struct CannotStabilizeDeprecated {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes_invalid_deprecation_version)]
pub struct InvalidDeprecationVersion {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes_item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes_missing_stability_attr)]
pub struct MissingStabilityAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_missing_const_stab_attr)]
pub struct MissingConstStabAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes_trait_impl_const_stable)]
#[note]
pub struct TraitImplConstStable {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_feature_only_on_nightly, code = "E0554")]
pub struct FeatureOnlyOnNightly {
    #[primary_span]
    pub span: Span,
    pub release_channel: &'static str,
}

#[derive(Diagnostic)]
#[diag(passes_unknown_feature, code = "E0635")]
pub struct UnknownFeature {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_implied_feature_not_exist)]
pub struct ImpliedFeatureNotExist {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub implied_by: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes_duplicate_feature_err, code = "E0636")]
pub struct DuplicateFeatureErr {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
}
#[derive(Diagnostic)]
#[diag(passes_missing_const_err)]
pub struct MissingConstErr {
    #[primary_span]
    #[help]
    pub fn_sig_span: Span,
    #[label]
    pub const_span: Span,
}

#[derive(LintDiagnostic)]
pub enum MultipleDeadCodes<'tcx> {
    #[diag(passes_dead_codes)]
    DeadCodes {
        multiple: bool,
        num: usize,
        descr: &'tcx str,
        participle: &'tcx str,
        name_list: DiagnosticSymbolList,
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
        name_list: DiagnosticSymbolList,
        #[subdiagnostic]
        change_fields_suggestion: ChangeFieldsToBeOfUnitType,
        #[subdiagnostic]
        parent_info: Option<ParentInfo<'tcx>>,
        #[subdiagnostic]
        ignored_derived_impls: Option<IgnoredDerivedImpls>,
    },
}

#[derive(Subdiagnostic)]
#[label(passes_parent_info)]
pub struct ParentInfo<'tcx> {
    pub num: usize,
    pub descr: &'tcx str,
    pub parent_descr: &'tcx str,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[note(passes_ignored_derived_impls)]
pub struct IgnoredDerivedImpls {
    pub name: Symbol,
    pub trait_list: DiagnosticSymbolList,
    pub trait_list_len: usize,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(passes_change_fields_to_be_of_unit_type, applicability = "has-placeholders")]
pub struct ChangeFieldsToBeOfUnitType {
    pub num: usize,
    #[suggestion_part(code = "()")]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes_proc_macro_typeerror)]
#[note]
pub(crate) struct ProcMacroTypeError<'tcx> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub found: Ty<'tcx>,
    pub kind: ProcMacroKind,
    pub expected_signature: &'static str,
}

#[derive(Diagnostic)]
#[diag(passes_proc_macro_diff_arg_count)]
pub(crate) struct ProcMacroDiffArguments {
    #[primary_span]
    #[label]
    pub span: Span,
    pub count: usize,
    pub kind: ProcMacroKind,
    pub expected_signature: &'static str,
}

#[derive(Diagnostic)]
#[diag(passes_proc_macro_missing_args)]
pub(crate) struct ProcMacroMissingArguments {
    #[primary_span]
    #[label]
    pub span: Span,
    pub expected_input_count: usize,
    pub kind: ProcMacroKind,
    pub expected_signature: &'static str,
}

#[derive(Diagnostic)]
#[diag(passes_proc_macro_invalid_abi)]
pub(crate) struct ProcMacroInvalidAbi {
    #[primary_span]
    pub span: Span,
    pub abi: &'static str,
}

#[derive(Diagnostic)]
#[diag(passes_proc_macro_unsafe)]
pub(crate) struct ProcMacroUnsafe {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes_skipping_const_checks)]
pub struct SkippingConstChecks {
    #[primary_span]
    pub span: Span,
}
