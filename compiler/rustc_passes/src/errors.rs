use std::{
    io::Error,
    path::{Path, PathBuf},
};

use rustc_ast::Label;
use rustc_errors::{error_code, Applicability, ErrorGuaranteed, IntoDiagnostic, MultiSpan};
use rustc_hir::{self as hir, ExprKind, Target};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{MainDefinition, Ty};
use rustc_span::{Span, Symbol, DUMMY_SP};

use crate::lang_items::Duplicate;

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

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
#[diag(passes::no_coverage_not_coverable, code = "E0788")]
pub struct IgnoredNoCoverageNotCoverable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::should_be_applied_to_fn)]
pub struct AttrShouldBeAppliedToFn {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::naked_tracked_caller, code = "E0736")]
pub struct NakedTrackedCaller {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::should_be_applied_to_fn, code = "E0739")]
pub struct TrackedCallerWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::should_be_applied_to_struct_enum, code = "E0701")]
pub struct NonExhaustiveWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
#[diag(passes::should_be_applied_to_static)]
pub struct AttrShouldBeAppliedToStatic {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::doc_expect_str)]
pub struct DocExpectStr<'a> {
    #[primary_span]
    pub attr_span: Span,
    pub attr_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::doc_alias_empty)]
pub struct DocAliasEmpty<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::doc_alias_bad_char)]
pub struct DocAliasBadChar<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub char_: char,
}

#[derive(Diagnostic)]
#[diag(passes::doc_alias_start_end)]
pub struct DocAliasStartEnd<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::doc_alias_bad_location)]
pub struct DocAliasBadLocation<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub location: &'a str,
}

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
#[diag(passes::doc_alias_not_string_literal)]
pub struct DocAliasNotStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::doc_alias_malformed)]
pub struct DocAliasMalformed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::doc_keyword_empty_mod)]
pub struct DocKeywordEmptyMod {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::doc_keyword_not_mod)]
pub struct DocKeywordNotMod {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::doc_keyword_invalid_ident)]
pub struct DocKeywordInvalidIdent {
    #[primary_span]
    pub span: Span,
    pub doc_keyword: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::doc_fake_variadic_not_valid)]
pub struct DocFakeVariadicNotValid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::doc_keyword_only_impl)]
pub struct DocKeywordOnlyImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
#[diag(passes::pass_by_value)]
pub struct PassByValue {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::allow_incoherent_impl)]
pub struct AllowIncoherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
#[diag(passes::no_link)]
pub struct NoLink {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::export_name)]
pub struct ExportName {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_layout_scalar_valid_range_not_struct)]
pub struct RustcLayoutScalarValidRangeNotStruct {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_layout_scalar_valid_range_arg)]
pub struct RustcLayoutScalarValidRangeArg {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_legacy_const_generics_only)]
pub struct RustcLegacyConstGenericsOnly {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub param_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_legacy_const_generics_index)]
pub struct RustcLegacyConstGenericsIndex {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub generics_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_legacy_const_generics_index_exceed)]
pub struct RustcLegacyConstGenericsIndexExceed {
    #[primary_span]
    #[label]
    pub span: Span,
    pub arg_count: usize,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_legacy_const_generics_index_negative)]
pub struct RustcLegacyConstGenericsIndexNegative {
    #[primary_span]
    pub invalid_args: Vec<Span>,
}

#[derive(Diagnostic)]
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
    #[suggestion(code = "", applicability = "machine-applicable")]
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

#[derive(Diagnostic)]
#[diag(passes::repr_ident, code = "E0565")]
pub struct ReprIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::repr_conflicting, code = "E0566")]
pub struct ReprConflicting;

#[derive(Diagnostic)]
#[diag(passes::used_static)]
pub struct UsedStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::used_compiler_linker)]
pub struct UsedCompilerLinker {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes::allow_internal_unstable)]
pub struct AllowInternalUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::debug_visualizer_placement)]
pub struct DebugVisualizerPlacement {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::debug_visualizer_invalid)]
#[note(passes::note_1)]
#[note(passes::note_2)]
#[note(passes::note_3)]
pub struct DebugVisualizerInvalid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::debug_visualizer_unreadable)]
pub struct DebugVisualizerUnreadable<'a> {
    #[primary_span]
    pub span: Span,
    pub file: &'a Path,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_allow_const_fn_unstable)]
pub struct RustcAllowConstFnUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_std_internal_symbol)]
pub struct RustcStdInternalSymbol {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::const_trait)]
pub struct ConstTrait {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::link_ordinal)]
pub struct LinkOrdinal {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
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

#[derive(Subdiagnostic)]
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
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub attr_span: Span,
    #[subdiagnostic]
    pub note: UnusedNote,
}

#[derive(Diagnostic)]
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

#[derive(Diagnostic)]
#[diag(passes::unused_multiple)]
pub struct UnusedMultiple {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_lint_opt_ty)]
pub struct RustcLintOptTy {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::rustc_lint_opt_deny_field_access)]
pub struct RustcLintOptDenyFieldAccess {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::collapse_debuginfo)]
pub struct CollapseDebuginfo {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::deprecated_annotation_has_no_effect)]
pub struct DeprecatedAnnotationHasNoEffect {
    #[suggestion(applicability = "machine-applicable", code = "")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::unknown_external_lang_item, code = "E0264")]
pub struct UnknownExternLangItem {
    #[primary_span]
    pub span: Span,
    pub lang_item: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::missing_panic_handler)]
pub struct MissingPanicHandler;

#[derive(Diagnostic)]
#[diag(passes::alloc_func_required)]
pub struct AllocFuncRequired;

#[derive(Diagnostic)]
#[diag(passes::missing_alloc_error_handler)]
pub struct MissingAllocErrorHandler;

#[derive(Diagnostic)]
#[diag(passes::missing_lang_item)]
#[note]
#[help]
pub struct MissingLangItem {
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::lang_item_on_incorrect_target, code = "E0718")]
pub struct LangItemOnIncorrectTarget {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
    pub expected_target: Target,
    pub actual_target: Target,
}

#[derive(Diagnostic)]
#[diag(passes::unknown_lang_item, code = "E0522")]
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
    fn into_diagnostic(
        self,
        handler: &'_ rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag =
            handler.struct_err(rustc_errors::fluent::passes::invalid_attr_at_crate_level);
        diag.set_span(self.span);
        diag.set_arg("name", self.name);
        // Only emit an error with a suggestion if we can create a string out
        // of the attribute span
        if let Some(src) = self.snippet {
            let replacement = src.replace("#!", "#");
            diag.span_suggestion_verbose(
                self.span,
                rustc_errors::fluent::passes::suggestion,
                replacement,
                rustc_errors::Applicability::MachineApplicable,
            );
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(passes::duplicate_diagnostic_item)]
pub struct DuplicateDiagnosticItem {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::duplicate_diagnostic_item_in_crate)]
pub struct DuplicateDiagnosticItemInCrate {
    #[note(passes::diagnostic_item_first_defined)]
    pub span: Option<Span>,
    pub orig_crate_name: Symbol,
    #[note]
    pub have_orig_crate_name: Option<()>,
    pub crate_name: Symbol,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::abi)]
pub struct Abi {
    #[primary_span]
    pub span: Span,
    pub abi: String,
}

#[derive(Diagnostic)]
#[diag(passes::align)]
pub struct Align {
    #[primary_span]
    pub span: Span,
    pub align: String,
}

#[derive(Diagnostic)]
#[diag(passes::size)]
pub struct Size {
    #[primary_span]
    pub span: Span,
    pub size: String,
}

#[derive(Diagnostic)]
#[diag(passes::homogeneous_aggregate)]
pub struct HomogeneousAggregate {
    #[primary_span]
    pub span: Span,
    pub homogeneous_aggregate: String,
}

#[derive(Diagnostic)]
#[diag(passes::layout_of)]
pub struct LayoutOf {
    #[primary_span]
    pub span: Span,
    pub normalized_ty: String,
    pub ty_layout: String,
}

#[derive(Diagnostic)]
#[diag(passes::unrecognized_field)]
pub struct UnrecognizedField {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::feature_stable_twice, code = "E0711")]
pub struct FeatureStableTwice {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub since: Symbol,
    pub prev_since: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::feature_previously_declared, code = "E0711")]
pub struct FeaturePreviouslyDeclared<'a, 'b> {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub declared: &'a str,
    pub prev_declared: &'b str,
}

#[derive(Diagnostic)]
#[diag(passes::expr_not_allowed_in_context, code = "E0744")]
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
    fn into_diagnostic(
        self,
        handler: &rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_span_err_with_code(
            self.span,
            rustc_errors::fluent::passes::break_non_loop,
            error_code!(E0571),
        );
        diag.set_arg("kind", self.kind);
        diag.span_label(self.span, rustc_errors::fluent::passes::label);
        if let Some(head) = self.head {
            diag.span_label(head, rustc_errors::fluent::passes::label2);
        }
        diag.span_suggestion(
            self.span,
            rustc_errors::fluent::passes::suggestion,
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
                        rustc_errors::fluent::passes::break_expr_suggestion,
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
#[diag(passes::continue_labeled_block, code = "E0696")]
pub struct ContinueLabeledBlock {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes::block_label)]
    pub block_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::break_inside_closure, code = "E0267")]
pub struct BreakInsideClosure<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes::closure_label)]
    pub closure_span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::break_inside_async_block, code = "E0267")]
pub struct BreakInsideAsyncBlock<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes::async_block_label)]
    pub closure_span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::outside_loop, code = "E0268")]
pub struct OutsideLoop<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::unlabeled_in_labeled_block, code = "E0695")]
pub struct UnlabeledInLabeledBlock<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub cf_type: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::unlabeled_cf_in_while_condition, code = "E0590")]
pub struct UnlabeledCfInWhileCondition<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub cf_type: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::cannot_inline_naked_function)]
pub struct CannotInlineNakedFunction {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(passes::undefined_naked_function_abi)]
pub struct UndefinedNakedFunctionAbi;

#[derive(Diagnostic)]
#[diag(passes::no_patterns)]
pub struct NoPatterns {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::params_not_allowed)]
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
    fn into_diagnostic(
        self,
        handler: &rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_span_err_with_code(
            self.span,
            rustc_errors::fluent::passes::naked_functions_asm_block,
            error_code!(E0787),
        );
        for span in self.multiple_asms.iter() {
            diag.span_label(*span, rustc_errors::fluent::passes::label_multiple_asm);
        }
        for span in self.non_asms.iter() {
            diag.span_label(*span, rustc_errors::fluent::passes::label_non_asm);
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(passes::naked_functions_operands, code = "E0787")]
pub struct NakedFunctionsOperands {
    #[primary_span]
    pub unsupported_operands: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(passes::naked_functions_asm_options, code = "E0787")]
pub struct NakedFunctionsAsmOptions {
    #[primary_span]
    pub span: Span,
    pub unsupported_options: String,
}

#[derive(Diagnostic)]
#[diag(passes::naked_functions_must_use_noreturn, code = "E0787")]
pub struct NakedFunctionsMustUseNoreturn {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = ", options(noreturn)", applicability = "machine-applicable")]
    pub last_span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::attr_only_on_main)]
pub struct AttrOnlyOnMain {
    #[primary_span]
    pub span: Span,
    pub attr: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::attr_only_on_root_main)]
pub struct AttrOnlyOnRootMain {
    #[primary_span]
    pub span: Span,
    pub attr: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::attr_only_in_functions)]
pub struct AttrOnlyInFunctions {
    #[primary_span]
    pub span: Span,
    pub attr: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::multiple_rustc_main, code = "E0137")]
pub struct MultipleRustcMain {
    #[primary_span]
    pub span: Span,
    #[label(passes::first)]
    pub first: Span,
    #[label(passes::additional)]
    pub additional: Span,
}

#[derive(Diagnostic)]
#[diag(passes::multiple_start_functions, code = "E0138")]
pub struct MultipleStartFunctions {
    #[primary_span]
    pub span: Span,
    #[label]
    pub labeled: Span,
    #[label(passes::previous)]
    pub previous: Span,
}

#[derive(Diagnostic)]
#[diag(passes::extern_main)]
pub struct ExternMain {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::unix_sigpipe_values)]
pub struct UnixSigpipeValues {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::no_main_function, code = "E0601")]
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
    fn into_diagnostic(
        self,
        handler: &'a rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'a, ErrorGuaranteed> {
        let mut diag = handler.struct_span_err_with_code(
            DUMMY_SP,
            rustc_errors::fluent::passes::no_main_function,
            error_code!(E0601),
        );
        diag.set_arg("crate_name", self.crate_name);
        diag.set_arg("filename", self.filename);
        diag.set_arg("has_filename", self.has_filename);
        let note = if !self.non_main_fns.is_empty() {
            for &span in &self.non_main_fns {
                diag.span_note(span, rustc_errors::fluent::passes::here_is_main);
            }
            diag.note(rustc_errors::fluent::passes::one_or_more_possible_main);
            diag.help(rustc_errors::fluent::passes::consider_moving_main);
            // There were some functions named `main` though. Try to give the user a hint.
            rustc_errors::fluent::passes::main_must_be_defined_at_crate
        } else if self.has_filename {
            rustc_errors::fluent::passes::consider_adding_main_to_file
        } else {
            rustc_errors::fluent::passes::consider_adding_main_at_crate
        };
        if self.file_empty {
            diag.note(note);
        } else {
            diag.set_span(self.sp.shrink_to_hi());
            diag.span_label(self.sp.shrink_to_hi(), note);
        }

        if let Some(main_def) = self.main_def_opt && main_def.opt_fn_def_id().is_none(){
            // There is something at `crate::main`, but it is not a function definition.
            diag.span_label(main_def.span, rustc_errors::fluent::passes::non_function_main);
        }

        if self.add_teach_note {
            diag.note(rustc_errors::fluent::passes::teach_note);
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
    fn into_diagnostic(
        self,
        handler: &rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err_with_code(
            match self.duplicate {
                Duplicate::Plain => rustc_errors::fluent::passes::duplicate_lang_item,

                Duplicate::Crate => rustc_errors::fluent::passes::duplicate_lang_item_crate,
                Duplicate::CrateDepends => {
                    rustc_errors::fluent::passes::duplicate_lang_item_crate_depends
                }
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
            diag.span_note(span, rustc_errors::fluent::passes::first_defined_span);
        } else {
            if self.orig_dependency_of.is_empty() {
                diag.note(rustc_errors::fluent::passes::first_defined_crate);
            } else {
                diag.note(rustc_errors::fluent::passes::first_defined_crate_depends);
            }

            if self.orig_is_local {
                diag.note(rustc_errors::fluent::passes::first_definition_local);
            } else {
                diag.note(rustc_errors::fluent::passes::first_definition_path);
            }

            if self.is_local {
                diag.note(rustc_errors::fluent::passes::second_definition_local);
            } else {
                diag.note(rustc_errors::fluent::passes::second_definition_path);
            }
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(passes::incorrect_target, code = "E0718")]
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
#[diag(passes::useless_assignment)]
pub struct UselessAssignment<'a> {
    pub is_field_assign: bool,
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(passes::only_has_effect_on)]
pub struct OnlyHasEffectOn {
    pub attr_name: Symbol,
    pub target_name: String,
}

#[derive(Diagnostic)]
#[diag(passes::object_lifetime_err)]
pub struct ObjectLifetimeErr {
    #[primary_span]
    pub span: Span,
    pub repr: String,
}

#[derive(Diagnostic)]
#[diag(passes::unrecognized_repr_hint, code = "E0552")]
#[help]
pub struct UnrecognizedReprHint {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
pub enum AttrApplication {
    #[diag(passes::attr_application_enum, code = "E0517")]
    Enum {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes::attr_application_struct, code = "E0517")]
    Struct {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes::attr_application_struct_union, code = "E0517")]
    StructUnion {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes::attr_application_struct_enum_union, code = "E0517")]
    StructEnumUnion {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
    #[diag(passes::attr_application_struct_enum_function_union, code = "E0517")]
    StructEnumFunctionUnion {
        #[primary_span]
        hint_span: Span,
        #[label]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(passes::transparent_incompatible, code = "E0692")]
pub struct TransparentIncompatible {
    #[primary_span]
    pub hint_spans: Vec<Span>,
    pub target: String,
}

#[derive(Diagnostic)]
#[diag(passes::deprecated_attribute, code = "E0549")]
pub struct DeprecatedAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::useless_stability)]
pub struct UselessStability {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes::item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes::invalid_stability)]
pub struct InvalidStability {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes::item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes::cannot_stabilize_deprecated)]
pub struct CannotStabilizeDeprecated {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes::item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes::invalid_deprecation_version)]
pub struct InvalidDeprecationVersion {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(passes::item)]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag(passes::missing_stability_attr)]
pub struct MissingStabilityAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::missing_const_stab_attr)]
pub struct MissingConstStabAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag(passes::trait_impl_const_stable)]
#[note]
pub struct TraitImplConstStable {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(passes::feature_only_on_nightly, code = "E0554")]
pub struct FeatureOnlyOnNightly {
    #[primary_span]
    pub span: Span,
    pub release_channel: &'static str,
}

#[derive(Diagnostic)]
#[diag(passes::unknown_feature, code = "E0635")]
pub struct UnknownFeature {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::implied_feature_not_exist)]
pub struct ImpliedFeatureNotExist {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub implied_by: Symbol,
}

#[derive(Diagnostic)]
#[diag(passes::duplicate_feature_err, code = "E0636")]
pub struct DuplicateFeatureErr {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
}
#[derive(Diagnostic)]
#[diag(passes::missing_const_err)]
pub struct MissingConstErr {
    #[primary_span]
    #[help]
    pub fn_sig_span: Span,
    #[label]
    pub const_span: Span,
}
