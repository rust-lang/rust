use rustc_errors::{
    AddToDiagnostic, DiagnosticBuilder, EmissionGuarantee, Handler, IntoDiagnostic, MultiSpan,
    SingleLabelManySpans,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag(builtin_macros_requires_cfg_pattern)]
pub(crate) struct RequiresCfgPattern {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_expected_one_cfg_pattern)]
pub(crate) struct OneCfgPattern {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_alloc_error_must_be_fn)]
pub(crate) struct AllocErrorMustBeFn {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_assert_requires_boolean)]
pub(crate) struct AssertRequiresBoolean {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_assert_requires_expression)]
pub(crate) struct AssertRequiresExpression {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub(crate) token: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_assert_missing_comma)]
pub(crate) struct AssertMissingComma {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(code = ", ", applicability = "maybe-incorrect", style = "short")]
    pub(crate) comma: Span,
}

#[derive(Diagnostic)]
pub(crate) enum CfgAccessibleInvalid {
    #[diag(builtin_macros_cfg_accessible_unspecified_path)]
    UnspecifiedPath(#[primary_span] Span),
    #[diag(builtin_macros_cfg_accessible_multiple_paths)]
    MultiplePaths(#[primary_span] Span),
    #[diag(builtin_macros_cfg_accessible_literal_path)]
    LiteralPath(#[primary_span] Span),
    #[diag(builtin_macros_cfg_accessible_has_args)]
    HasArguments(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(builtin_macros_cfg_accessible_indeterminate)]
pub(crate) struct CfgAccessibleIndeterminate {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_missing_literal)]
#[note]
pub(crate) struct ConcatMissingLiteral {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_bytestr)]
pub(crate) struct ConcatBytestr {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_c_str_lit)]
pub(crate) struct ConcatCStrLit {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_export_macro_rules)]
pub(crate) struct ExportMacroRules {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_proc_macro)]
pub(crate) struct ProcMacro {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_invalid_crate_attribute)]
pub(crate) struct InvalidCrateAttr {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_non_abi)]
pub(crate) struct NonABI {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_trace_macros)]
pub(crate) struct TraceMacros {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_bench_sig)]
pub(crate) struct BenchSig {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_test_arg_non_lifetime)]
pub(crate) struct TestArgNonLifetime {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_should_panic)]
pub(crate) struct ShouldPanic {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_test_args)]
pub(crate) struct TestArgs {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_alloc_must_statics)]
pub(crate) struct AllocMustStatics {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_bytes_invalid)]
pub(crate) struct ConcatBytesInvalid {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) lit_kind: &'static str,
    #[subdiagnostic]
    pub(crate) sugg: Option<ConcatBytesInvalidSuggestion>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ConcatBytesInvalidSuggestion {
    #[suggestion(
        builtin_macros_byte_char,
        code = "b{snippet}",
        applicability = "machine-applicable"
    )]
    CharLit {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[suggestion(
        builtin_macros_byte_str,
        code = "b{snippet}",
        applicability = "machine-applicable"
    )]
    StrLit {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[suggestion(
        builtin_macros_number_array,
        code = "[{snippet}]",
        applicability = "machine-applicable"
    )]
    IntLit {
        #[primary_span]
        span: Span,
        snippet: String,
    },
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_bytes_oob)]
pub(crate) struct ConcatBytesOob {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_bytes_non_u8)]
pub(crate) struct ConcatBytesNonU8 {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_bytes_missing_literal)]
#[note]
pub(crate) struct ConcatBytesMissingLiteral {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_bytes_array)]
pub(crate) struct ConcatBytesArray {
    #[primary_span]
    pub(crate) span: Span,
    #[note]
    #[help]
    pub(crate) bytestr: bool,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_bytes_bad_repeat)]
pub(crate) struct ConcatBytesBadRepeat {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_idents_missing_args)]
pub(crate) struct ConcatIdentsMissingArgs {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_idents_missing_comma)]
pub(crate) struct ConcatIdentsMissingComma {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_concat_idents_ident_args)]
pub(crate) struct ConcatIdentsIdentArgs {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_bad_derive_target, code = "E0774")]
pub(crate) struct BadDeriveTarget {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(builtin_macros_label2)]
    pub(crate) item: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_tests_not_support)]
pub(crate) struct TestsNotSupport {}

#[derive(Diagnostic)]
#[diag(builtin_macros_unexpected_lit, code = "E0777")]
pub(crate) struct BadDeriveLit {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub help: BadDeriveLitHelp,
}

#[derive(Subdiagnostic)]
pub(crate) enum BadDeriveLitHelp {
    #[help(builtin_macros_str_lit)]
    StrLit { sym: Symbol },
    #[help(builtin_macros_other)]
    Other,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_derive_path_args_list)]
pub(crate) struct DerivePathArgsList {
    #[suggestion(code = "", applicability = "machine-applicable")]
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_derive_path_args_value)]
pub(crate) struct DerivePathArgsValue {
    #[suggestion(code = "", applicability = "machine-applicable")]
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_no_default_variant)]
#[help]
pub(crate) struct NoDefaultVariant {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) suggs: Vec<NoDefaultVariantSugg>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    builtin_macros_suggestion,
    code = "#[default] {ident}",
    applicability = "maybe-incorrect",
    style = "tool-only"
)]
pub(crate) struct NoDefaultVariantSugg {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_multiple_defaults)]
#[note]
pub(crate) struct MultipleDefaults {
    #[primary_span]
    pub(crate) span: Span,
    #[label]
    pub(crate) first: Span,
    #[label(builtin_macros_additional)]
    pub additional: Vec<Span>,
    #[subdiagnostic]
    pub suggs: Vec<MultipleDefaultsSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    builtin_macros_suggestion,
    applicability = "maybe-incorrect",
    style = "tool-only"
)]
pub(crate) struct MultipleDefaultsSugg {
    #[suggestion_part(code = "")]
    pub(crate) spans: Vec<Span>,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_non_unit_default)]
#[help]
pub(crate) struct NonUnitDefault {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_non_exhaustive_default)]
#[help]
pub(crate) struct NonExhaustiveDefault {
    #[primary_span]
    pub(crate) span: Span,
    #[label]
    pub(crate) non_exhaustive: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_multiple_default_attrs)]
#[note]
pub(crate) struct MultipleDefaultAttrs {
    #[primary_span]
    pub(crate) span: Span,
    #[label]
    pub(crate) first: Span,
    #[label(builtin_macros_label_again)]
    pub(crate) first_rest: Span,
    #[help]
    pub(crate) rest: MultiSpan,
    pub(crate) only_one: bool,
    #[subdiagnostic]
    pub(crate) sugg: MultipleDefaultAttrsSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    builtin_macros_help,
    applicability = "machine-applicable",
    style = "tool-only"
)]
pub(crate) struct MultipleDefaultAttrsSugg {
    #[suggestion_part(code = "")]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_default_arg)]
pub(crate) struct DefaultHasArg {
    #[primary_span]
    #[suggestion(code = "#[default]", style = "hidden", applicability = "maybe-incorrect")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_derive_macro_call)]
pub(crate) struct DeriveMacroCall {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_cannot_derive_union)]
pub(crate) struct DeriveUnion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_env_takes_args)]
pub(crate) struct EnvTakesArgs {
    #[primary_span]
    pub(crate) span: Span,
}

pub(crate) struct EnvNotDefinedWithUserMessage {
    pub(crate) span: Span,
    pub(crate) msg_from_user: Symbol,
}

// Hand-written implementation to support custom user messages.
impl<'a, G: EmissionGuarantee> IntoDiagnostic<'a, G> for EnvNotDefinedWithUserMessage {
    #[track_caller]
    fn into_diagnostic(self, handler: &'a Handler) -> DiagnosticBuilder<'a, G> {
        #[expect(
            rustc::untranslatable_diagnostic,
            reason = "cannot translate user-provided messages"
        )]
        let mut diag = handler.struct_diagnostic(self.msg_from_user.to_string());
        diag.set_span(self.span);
        diag
    }
}

#[derive(Diagnostic)]
pub(crate) enum EnvNotDefined<'a> {
    #[diag(builtin_macros_env_not_defined)]
    #[help(builtin_macros_cargo)]
    CargoEnvVar {
        #[primary_span]
        span: Span,
        var: Symbol,
        var_expr: &'a rustc_ast::Expr,
    },
    #[diag(builtin_macros_env_not_defined)]
    #[help(builtin_macros_custom)]
    CustomEnvVar {
        #[primary_span]
        span: Span,
        var: Symbol,
        var_expr: &'a rustc_ast::Expr,
    },
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_requires_string)]
pub(crate) struct FormatRequiresString {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_duplicate_arg)]
pub(crate) struct FormatDuplicateArg {
    #[primary_span]
    pub(crate) span: Span,
    #[label(builtin_macros_label1)]
    pub(crate) prev: Span,
    #[label(builtin_macros_label2)]
    pub(crate) duplicate: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_positional_after_named)]
pub(crate) struct PositionalAfterNamed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(builtin_macros_named_args)]
    pub(crate) args: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_string_invalid)]
pub(crate) struct InvalidFormatString {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) desc: String,
    pub(crate) label1: String,
    #[subdiagnostic]
    pub(crate) note_: Option<InvalidFormatStringNote>,
    #[subdiagnostic]
    pub(crate) label_: Option<InvalidFormatStringLabel>,
    #[subdiagnostic]
    pub(crate) sugg_: Option<InvalidFormatStringSuggestion>,
}

#[derive(Subdiagnostic)]
#[note(builtin_macros_note)]
pub(crate) struct InvalidFormatStringNote {
    pub(crate) note: String,
}

#[derive(Subdiagnostic)]
#[label(builtin_macros_second_label)]
pub(crate) struct InvalidFormatStringLabel {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) label: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    builtin_macros_sugg,
    style = "verbose",
    applicability = "machine-applicable"
)]
pub(crate) struct InvalidFormatStringSuggestion {
    #[suggestion_part(code = "{len}")]
    pub(crate) captured: Span,
    pub(crate) len: String,
    #[suggestion_part(code = ", {arg}")]
    pub(crate) span: Span,
    pub(crate) arg: String,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_no_arg_named)]
#[note]
#[note(builtin_macros_note2)]
pub(crate) struct FormatNoArgNamed {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_unknown_trait)]
#[note]
pub(crate) struct FormatUnknownTrait<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ty: &'a str,
    #[subdiagnostic]
    pub(crate) suggs: Vec<FormatUnknownTraitSugg>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    builtin_macros_suggestion,
    code = "{fmt}",
    style = "tool-only",
    applicability = "maybe-incorrect"
)]
pub struct FormatUnknownTraitSugg {
    #[primary_span]
    pub span: Span,
    pub fmt: &'static str,
    pub trait_name: &'static str,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_unused_arg)]
pub(crate) struct FormatUnusedArg {
    #[primary_span]
    #[label(builtin_macros_format_unused_arg)]
    pub(crate) span: Span,
    pub(crate) named: bool,
}

// Allow the singular form to be a subdiagnostic of the multiple-unused
// form of diagnostic.
impl AddToDiagnostic for FormatUnusedArg {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, f: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        diag.set_arg("named", self.named);
        let msg = f(diag, crate::fluent_generated::builtin_macros_format_unused_arg.into());
        diag.span_label(self.span, msg);
    }
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_unused_args)]
pub(crate) struct FormatUnusedArgs {
    #[primary_span]
    pub(crate) unused: Vec<Span>,
    #[label]
    pub(crate) fmt: Span,
    #[subdiagnostic]
    pub(crate) unused_labels: Vec<FormatUnusedArg>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_format_pos_mismatch)]
pub(crate) struct FormatPositionalMismatch {
    #[primary_span]
    pub(crate) span: MultiSpan,
    pub(crate) n: usize,
    pub(crate) desc: String,
    #[subdiagnostic]
    pub(crate) highlight: SingleLabelManySpans,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_test_case_non_item)]
pub(crate) struct TestCaseNonItem {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_test_bad_fn)]
pub(crate) struct TestBadFn {
    #[primary_span]
    pub(crate) span: Span,
    #[label]
    pub(crate) cause: Span,
    pub(crate) kind: &'static str,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_explicit_register_name)]
pub(crate) struct AsmExplicitRegisterName {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_mutually_exclusive)]
pub(crate) struct AsmMutuallyExclusive {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
    pub(crate) opt1: &'static str,
    pub(crate) opt2: &'static str,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_pure_combine)]
pub(crate) struct AsmPureCombine {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_pure_no_output)]
pub(crate) struct AsmPureNoOutput {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_modifier_invalid)]
pub(crate) struct AsmModifierInvalid {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_requires_template)]
pub(crate) struct AsmRequiresTemplate {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_expected_comma)]
pub(crate) struct AsmExpectedComma {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_underscore_input)]
pub(crate) struct AsmUnderscoreInput {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_sym_no_path)]
pub(crate) struct AsmSymNoPath {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_expected_other)]
pub(crate) struct AsmExpectedOther {
    #[primary_span]
    #[label(builtin_macros_asm_expected_other)]
    pub(crate) span: Span,
    pub(crate) is_global_asm: bool,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_duplicate_arg)]
pub(crate) struct AsmDuplicateArg {
    #[primary_span]
    #[label(builtin_macros_arg)]
    pub(crate) span: Span,
    #[label]
    pub(crate) prev: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_pos_after)]
pub(crate) struct AsmPositionalAfter {
    #[primary_span]
    #[label(builtin_macros_pos)]
    pub(crate) span: Span,
    #[label(builtin_macros_named)]
    pub(crate) named: Vec<Span>,
    #[label(builtin_macros_explicit)]
    pub(crate) explicit: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_noreturn)]
pub(crate) struct AsmNoReturn {
    #[primary_span]
    pub(crate) outputs_sp: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_global_asm_clobber_abi)]
pub(crate) struct GlobalAsmClobberAbi {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

pub(crate) struct AsmClobberNoReg {
    pub(crate) spans: Vec<Span>,
    pub(crate) clobbers: Vec<Span>,
}

impl<'a, G: EmissionGuarantee> IntoDiagnostic<'a, G> for AsmClobberNoReg {
    fn into_diagnostic(self, handler: &'a Handler) -> DiagnosticBuilder<'a, G> {
        let mut diag =
            handler.struct_diagnostic(crate::fluent_generated::builtin_macros_asm_clobber_no_reg);
        diag.set_span(self.spans.clone());
        // eager translation as `span_labels` takes `AsRef<str>`
        let lbl1 = handler.eagerly_translate_to_string(
            crate::fluent_generated::builtin_macros_asm_clobber_abi,
            [].into_iter(),
        );
        diag.span_labels(self.clobbers, &lbl1);
        let lbl2 = handler.eagerly_translate_to_string(
            crate::fluent_generated::builtin_macros_asm_clobber_outputs,
            [].into_iter(),
        );
        diag.span_labels(self.spans, &lbl2);
        diag
    }
}

#[derive(Diagnostic)]
#[diag(builtin_macros_asm_opt_already_provided)]
pub(crate) struct AsmOptAlreadyprovided {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) symbol: Symbol,
    #[suggestion(code = "", applicability = "machine-applicable", style = "tool-only")]
    pub(crate) full_span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_test_runner_invalid)]
pub(crate) struct TestRunnerInvalid {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_test_runner_nargs)]
pub(crate) struct TestRunnerNargs {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_expected_register_class_or_explicit_register)]
pub(crate) struct ExpectedRegisterClassOrExplicitRegister {
    #[primary_span]
    pub(crate) span: Span,
}
