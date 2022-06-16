use rustc_errors::{Applicability, MultiSpan};
use rustc_macros::{LintDiagnostic, SessionDiagnostic};
use rustc_span::{Span, Symbol};

#[derive(LintDiagnostic)]
#[lint(passes::outer_crate_level_attr)]
pub struct OuterCrateLevelAttr;

#[derive(LintDiagnostic)]
#[lint(passes::inner_crate_level_attr)]
pub struct InnerCrateLevelAttr;

#[derive(LintDiagnostic)]
#[lint(passes::ignored_attr_with_macro)]
pub struct IgnoredAttrWithMacro<'a> {
    pub sym: &'a str,
}

#[derive(LintDiagnostic)]
#[lint(passes::ignored_attr)]
pub struct IgnoredAttr<'a> {
    pub sym: &'a str,
}

#[derive(LintDiagnostic)]
#[lint(passes::inline_ignored_function_prototype)]
pub struct IgnoredInlineAttrFnProto;

#[derive(LintDiagnostic)]
#[lint(passes::inline_ignored_constants)]
#[warn_]
#[note]
pub struct IgnoredInlineAttrConstants;

#[derive(SessionDiagnostic)]
#[error(passes::inline_not_fn_or_closure, code = "E0518")]
pub struct InlineNotFnOrClosure {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[lint(passes::no_coverage_ignored_function_prototype)]
pub struct IgnoredNoCoverageFnProto;

#[derive(LintDiagnostic)]
#[lint(passes::no_coverage_propagate)]
pub struct IgnoredNoCoveragePropagate;

#[derive(LintDiagnostic)]
#[lint(passes::no_coverage_fn_defn)]
pub struct IgnoredNoCoverageFnDefn;

#[derive(SessionDiagnostic)]
#[error(passes::no_coverage_not_coverable, code = "E0788")]
pub struct IgnoredNoCoverageNotCoverable {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::should_be_applied_to_fn)]
pub struct AttrShouldBeAppliedToFn {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::naked_tracked_caller, code = "E0736")]
pub struct NakedTrackedCaller {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::should_be_applied_to_fn, code = "E0739")]
pub struct TrackedCallerWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::should_be_applied_to_struct_enum, code = "E0701")]
pub struct NonExhaustiveWrongLocation {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::should_be_applied_to_trait)]
pub struct AttrShouldBeAppliedToTrait {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(LintDiagnostic)]
#[lint(passes::target_feature_on_statement)]
pub struct TargetFeatureOnStatement;

#[derive(SessionDiagnostic)]
#[error(passes::should_be_applied_to_static)]
pub struct AttrShouldBeAppliedToStatic {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub defn_span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_expect_str)]
pub struct DocExpectStr<'a> {
    #[primary_span]
    pub attr_span: Span,
    pub attr_name: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_alias_empty)]
pub struct DocAliasEmpty<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_alias_bad_char)]
pub struct DocAliasBadChar<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub char_: char,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_alias_start_end)]
pub struct DocAliasStartEnd<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_alias_bad_location)]
pub struct DocAliasBadLocation<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub location: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_alias_not_an_alias)]
pub struct DocAliasNotAnAlias<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(LintDiagnostic)]
#[lint(passes::doc_alias_duplicated)]
pub struct DocAliasDuplicated {
    #[label]
    pub first_defn: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_alias_not_string_literal)]
pub struct DocAliasNotStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_alias_malformed)]
pub struct DocAliasMalformed {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_keyword_empty_mod)]
pub struct DocKeywordEmptyMod {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_keyword_not_mod)]
pub struct DocKeywordNotMod {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_keyword_invalid_ident)]
pub struct DocKeywordInvalidIdent {
    #[primary_span]
    pub span: Span,
    pub doc_keyword: Symbol,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_fake_variadic_not_valid)]
pub struct DocFakeVariadicNotValid {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_keyword_only_impl)]
pub struct DocKeywordOnlyImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_inline_conflict)]
#[help]
pub struct DocKeywordConflict {
    #[primary_span]
    pub spans: MultiSpan,
}

#[derive(LintDiagnostic)]
#[lint(passes::doc_inline_only_use)]
#[note]
pub struct DocInlineOnlyUse {
    #[label]
    pub attr_span: Span,
    #[label(passes::not_a_use_item_label)]
    pub item_span: Option<Span>,
}

#[derive(SessionDiagnostic)]
#[error(passes::doc_attr_not_crate_level)]
pub struct DocAttrNotCrateLevel<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_name: &'a str,
}

#[derive(LintDiagnostic)]
#[lint(passes::doc_test_unknown)]
pub struct DocTestUnknown {
    pub path: String,
}

#[derive(LintDiagnostic)]
#[lint(passes::doc_test_takes_list)]
pub struct DocTestTakesList;

#[derive(LintDiagnostic)]
#[lint(passes::doc_primitive)]
pub struct DocPrimitive;

#[derive(LintDiagnostic)]
#[lint(passes::doc_test_unknown_any)]
pub struct DocTestUnknownAny {
    pub path: String,
}

#[derive(LintDiagnostic)]
#[lint(passes::doc_test_unknown_spotlight)]
#[note]
#[note(passes::no_op_note)]
pub struct DocTestUnknownSpotlight {
    pub path: String,
    #[suggestion_short(applicability = "machine-applicable", code = "notable_trait")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[lint(passes::doc_test_unknown_include)]
pub struct DocTestUnknownInclude {
    pub path: String,
    pub value: String,
    pub inner: &'static str,
    #[suggestion(code = "#{inner}[doc = include_str!(\"{value}\")]")]
    pub sugg: (Span, Applicability),
}

#[derive(LintDiagnostic)]
#[lint(passes::doc_invalid)]
pub struct DocInvalid;

#[derive(SessionDiagnostic)]
#[error(passes::pass_by_value)]
pub struct PassByValue {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::allow_incoherent_impl)]
pub struct AllowIncoherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(passes::has_incoherent_inherent_impl)]
pub struct HasIncoherentInherentImpl {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[lint(passes::must_use_async)]
pub struct MustUseAsync {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[lint(passes::must_use_no_effect)]
pub struct MustUseNoEffect {
    pub article: &'static str,
    pub target: rustc_hir::Target,
}

#[derive(SessionDiagnostic)]
#[error(passes::must_not_suspend)]
pub struct MustNotSuspend {
    #[primary_span]
    pub attr_span: Span,
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[lint(passes::cold)]
#[warn_]
pub struct Cold {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[lint(passes::link)]
#[warn_]
pub struct Link {
    #[label]
    pub span: Option<Span>,
}
