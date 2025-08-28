use std::num::{NonZero, ParseIntError};

use rustc_ast::token;
use rustc_ast::util::literal::LitError;
use rustc_errors::codes::*;
use rustc_errors::{
    Diag, DiagCtxtHandle, DiagMessage, Diagnostic, EmissionGuarantee, ErrorGuaranteed, Level,
    MultiSpan,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};
use rustc_target::spec::{SplitDebuginfo, StackProtector, TargetTuple};

use crate::config::CrateType;
use crate::parse::ParseSess;

#[derive(Diagnostic)]
pub(crate) enum AppleDeploymentTarget {
    #[diag(session_apple_deployment_target_invalid)]
    Invalid { env_var: &'static str, error: ParseIntError },
    #[diag(session_apple_deployment_target_too_low)]
    TooLow { env_var: &'static str, version: String, os_min: String },
}

pub(crate) struct FeatureGateError {
    pub(crate) span: MultiSpan,
    pub(crate) explain: DiagMessage,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for FeatureGateError {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        Diag::new(dcx, level, self.explain).with_span(self.span).with_code(E0658)
    }
}

#[derive(Subdiagnostic)]
#[note(session_feature_diagnostic_for_issue)]
pub(crate) struct FeatureDiagnosticForIssue {
    pub(crate) n: NonZero<u32>,
}

#[derive(Subdiagnostic)]
#[note(session_feature_suggest_upgrade_compiler)]
pub(crate) struct SuggestUpgradeCompiler {
    date: &'static str,
}

impl SuggestUpgradeCompiler {
    pub(crate) fn ui_testing() -> Self {
        Self { date: "YYYY-MM-DD" }
    }

    pub(crate) fn new() -> Option<Self> {
        let date = option_env!("CFG_VER_DATE")?;

        Some(Self { date })
    }
}

#[derive(Subdiagnostic)]
#[help(session_feature_diagnostic_help)]
pub(crate) struct FeatureDiagnosticHelp {
    pub(crate) feature: Symbol,
}

#[derive(Subdiagnostic)]
#[suggestion(
    session_feature_diagnostic_suggestion,
    applicability = "maybe-incorrect",
    code = "#![feature({feature})]\n"
)]
pub struct FeatureDiagnosticSuggestion {
    pub feature: Symbol,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[help(session_cli_feature_diagnostic_help)]
pub(crate) struct CliFeatureDiagnosticHelp {
    pub(crate) feature: Symbol,
}

#[derive(Diagnostic)]
#[diag(session_not_circumvent_feature)]
pub(crate) struct NotCircumventFeature;

#[derive(Diagnostic)]
#[diag(session_linker_plugin_lto_windows_not_supported)]
pub(crate) struct LinkerPluginToWindowsNotSupported;

#[derive(Diagnostic)]
#[diag(session_profile_use_file_does_not_exist)]
pub(crate) struct ProfileUseFileDoesNotExist<'a> {
    pub(crate) path: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session_profile_sample_use_file_does_not_exist)]
pub(crate) struct ProfileSampleUseFileDoesNotExist<'a> {
    pub(crate) path: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session_target_requires_unwind_tables)]
pub(crate) struct TargetRequiresUnwindTables;

#[derive(Diagnostic)]
#[diag(session_instrumentation_not_supported)]
pub(crate) struct InstrumentationNotSupported {
    pub(crate) us: String,
}

#[derive(Diagnostic)]
#[diag(session_sanitizer_not_supported)]
pub(crate) struct SanitizerNotSupported {
    pub(crate) us: String,
}

#[derive(Diagnostic)]
#[diag(session_sanitizers_not_supported)]
pub(crate) struct SanitizersNotSupported {
    pub(crate) us: String,
}

#[derive(Diagnostic)]
#[diag(session_cannot_mix_and_match_sanitizers)]
pub(crate) struct CannotMixAndMatchSanitizers {
    pub(crate) first: String,
    pub(crate) second: String,
}

#[derive(Diagnostic)]
#[diag(session_cannot_enable_crt_static_linux)]
pub(crate) struct CannotEnableCrtStaticLinux;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_requires_lto)]
pub(crate) struct SanitizerCfiRequiresLto;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_requires_single_codegen_unit)]
pub(crate) struct SanitizerCfiRequiresSingleCodegenUnit;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_canonical_jump_tables_requires_cfi)]
pub(crate) struct SanitizerCfiCanonicalJumpTablesRequiresCfi;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_generalize_pointers_requires_cfi)]
pub(crate) struct SanitizerCfiGeneralizePointersRequiresCfi;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_normalize_integers_requires_cfi)]
pub(crate) struct SanitizerCfiNormalizeIntegersRequiresCfi;

#[derive(Diagnostic)]
#[diag(session_sanitizer_kcfi_arity_requires_kcfi)]
pub(crate) struct SanitizerKcfiArityRequiresKcfi;

#[derive(Diagnostic)]
#[diag(session_sanitizer_kcfi_requires_panic_abort)]
pub(crate) struct SanitizerKcfiRequiresPanicAbort;

#[derive(Diagnostic)]
#[diag(session_split_lto_unit_requires_lto)]
pub(crate) struct SplitLtoUnitRequiresLto;

#[derive(Diagnostic)]
#[diag(session_unstable_virtual_function_elimination)]
pub(crate) struct UnstableVirtualFunctionElimination;

#[derive(Diagnostic)]
#[diag(session_unsupported_dwarf_version)]
#[help(session_unsupported_dwarf_version_help)]
pub(crate) struct UnsupportedDwarfVersion {
    pub(crate) dwarf_version: u32,
}

#[derive(Diagnostic)]
#[diag(session_embed_source_insufficient_dwarf_version)]
pub(crate) struct EmbedSourceInsufficientDwarfVersion {
    pub(crate) dwarf_version: u32,
}

#[derive(Diagnostic)]
#[diag(session_embed_source_requires_debug_info)]
pub(crate) struct EmbedSourceRequiresDebugInfo;

#[derive(Diagnostic)]
#[diag(session_target_stack_protector_not_supported)]
pub(crate) struct StackProtectorNotSupportedForTarget<'a> {
    pub(crate) stack_protector: StackProtector,
    pub(crate) target_triple: &'a TargetTuple,
}

#[derive(Diagnostic)]
#[diag(session_target_small_data_threshold_not_supported)]
pub(crate) struct SmallDataThresholdNotSupportedForTarget<'a> {
    pub(crate) target_triple: &'a TargetTuple,
}

#[derive(Diagnostic)]
#[diag(session_branch_protection_requires_aarch64)]
pub(crate) struct BranchProtectionRequiresAArch64;

#[derive(Diagnostic)]
#[diag(session_split_debuginfo_unstable_platform)]
pub(crate) struct SplitDebugInfoUnstablePlatform {
    pub(crate) debuginfo: SplitDebuginfo,
}

#[derive(Diagnostic)]
#[diag(session_file_is_not_writeable)]
pub(crate) struct FileIsNotWriteable<'a> {
    pub(crate) file: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session_file_write_fail)]
pub(crate) struct FileWriteFail<'a> {
    pub(crate) path: &'a std::path::Path,
    pub(crate) err: String,
}

#[derive(Diagnostic)]
#[diag(session_crate_name_empty)]
pub(crate) struct CrateNameEmpty {
    #[primary_span]
    pub(crate) span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(session_invalid_character_in_crate_name)]
pub(crate) struct InvalidCharacterInCrateName {
    #[primary_span]
    pub(crate) span: Option<Span>,
    pub(crate) character: char,
    pub(crate) crate_name: Symbol,
    #[help]
    pub(crate) help: Option<()>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(session_expr_parentheses_needed, applicability = "machine-applicable")]
pub struct ExprParenthesesNeeded {
    #[suggestion_part(code = "(")]
    left: Span,
    #[suggestion_part(code = ")")]
    right: Span,
}

impl ExprParenthesesNeeded {
    pub fn surrounding(s: Span) -> Self {
        ExprParenthesesNeeded { left: s.shrink_to_lo(), right: s.shrink_to_hi() }
    }
}

#[derive(Diagnostic)]
#[diag(session_skipping_const_checks)]
pub(crate) struct SkippingConstChecks {
    #[subdiagnostic]
    pub(crate) unleashed_features: Vec<UnleashedFeatureHelp>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnleashedFeatureHelp {
    #[help(session_unleashed_feature_help_named)]
    Named {
        #[primary_span]
        span: Span,
        gate: Symbol,
    },
    #[help(session_unleashed_feature_help_unnamed)]
    Unnamed {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(session_invalid_literal_suffix)]
struct InvalidLiteralSuffix<'a> {
    #[primary_span]
    #[label]
    span: Span,
    // FIXME(#100717)
    kind: &'a str,
    suffix: Symbol,
}

#[derive(Diagnostic)]
#[diag(session_invalid_int_literal_width)]
#[help]
struct InvalidIntLiteralWidth {
    #[primary_span]
    span: Span,
    width: String,
}

#[derive(Diagnostic)]
#[diag(session_invalid_num_literal_base_prefix)]
#[note]
struct InvalidNumLiteralBasePrefix {
    #[primary_span]
    #[suggestion(applicability = "maybe-incorrect", code = "{fixed}")]
    span: Span,
    fixed: String,
}

#[derive(Diagnostic)]
#[diag(session_invalid_num_literal_suffix)]
#[help]
struct InvalidNumLiteralSuffix {
    #[primary_span]
    #[label]
    span: Span,
    suffix: String,
}

#[derive(Diagnostic)]
#[diag(session_invalid_float_literal_width)]
#[help]
struct InvalidFloatLiteralWidth {
    #[primary_span]
    span: Span,
    width: String,
}

#[derive(Diagnostic)]
#[diag(session_invalid_float_literal_suffix)]
#[help]
struct InvalidFloatLiteralSuffix {
    #[primary_span]
    #[label]
    span: Span,
    suffix: String,
}

#[derive(Diagnostic)]
#[diag(session_int_literal_too_large)]
#[note]
struct IntLiteralTooLarge {
    #[primary_span]
    span: Span,
    limit: String,
}

#[derive(Diagnostic)]
#[diag(session_hexadecimal_float_literal_not_supported)]
struct HexadecimalFloatLiteralNotSupported {
    #[primary_span]
    #[label(session_not_supported)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(session_octal_float_literal_not_supported)]
struct OctalFloatLiteralNotSupported {
    #[primary_span]
    #[label(session_not_supported)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(session_binary_float_literal_not_supported)]
struct BinaryFloatLiteralNotSupported {
    #[primary_span]
    #[label(session_not_supported)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(session_unsupported_crate_type_for_target)]
pub(crate) struct UnsupportedCrateTypeForTarget<'a> {
    pub(crate) crate_type: CrateType,
    pub(crate) target_triple: &'a TargetTuple,
}

pub fn report_lit_error(
    psess: &ParseSess,
    err: LitError,
    lit: token::Lit,
    span: Span,
) -> ErrorGuaranteed {
    create_lit_error(psess, err, lit, span).emit()
}

pub fn create_lit_error(psess: &ParseSess, err: LitError, lit: token::Lit, span: Span) -> Diag<'_> {
    // Checks if `s` looks like i32 or u1234 etc.
    fn looks_like_width_suffix(first_chars: &[char], s: &str) -> bool {
        s.len() > 1 && s.starts_with(first_chars) && s[1..].chars().all(|c| c.is_ascii_digit())
    }

    // Try to lowercase the prefix if the prefix and suffix are valid.
    fn fix_base_capitalisation(prefix: &str, suffix: &str) -> Option<String> {
        let mut chars = suffix.chars();

        let base_char = chars.next().unwrap();
        let base = match base_char {
            'B' => 2,
            'O' => 8,
            'X' => 16,
            _ => return None,
        };

        // check that the suffix contains only base-appropriate characters
        let valid = prefix == "0"
            && chars
                .filter(|c| *c != '_')
                .take_while(|c| *c != 'i' && *c != 'u')
                .all(|c| c.to_digit(base).is_some());

        valid.then(|| format!("0{}{}", base_char.to_ascii_lowercase(), &suffix[1..]))
    }

    let dcx = psess.dcx();
    match err {
        LitError::InvalidSuffix(suffix) => {
            dcx.create_err(InvalidLiteralSuffix { span, kind: lit.kind.descr(), suffix })
        }
        LitError::InvalidIntSuffix(suffix) => {
            let suf = suffix.as_str();
            if looks_like_width_suffix(&['i', 'u'], suf) {
                // If it looks like a width, try to be helpful.
                dcx.create_err(InvalidIntLiteralWidth { span, width: suf[1..].into() })
            } else if let Some(fixed) = fix_base_capitalisation(lit.symbol.as_str(), suf) {
                dcx.create_err(InvalidNumLiteralBasePrefix { span, fixed })
            } else {
                dcx.create_err(InvalidNumLiteralSuffix { span, suffix: suf.to_string() })
            }
        }
        LitError::InvalidFloatSuffix(suffix) => {
            let suf = suffix.as_str();
            if looks_like_width_suffix(&['f'], suf) {
                // If it looks like a width, try to be helpful.
                dcx.create_err(InvalidFloatLiteralWidth { span, width: suf[1..].to_string() })
            } else {
                dcx.create_err(InvalidFloatLiteralSuffix { span, suffix: suf.to_string() })
            }
        }
        LitError::NonDecimalFloat(base) => match base {
            16 => dcx.create_err(HexadecimalFloatLiteralNotSupported { span }),
            8 => dcx.create_err(OctalFloatLiteralNotSupported { span }),
            2 => dcx.create_err(BinaryFloatLiteralNotSupported { span }),
            _ => unreachable!(),
        },
        LitError::IntTooLarge(base) => {
            let max = u128::MAX;
            let limit = match base {
                2 => format!("{max:#b}"),
                8 => format!("{max:#o}"),
                16 => format!("{max:#x}"),
                _ => format!("{max}"),
            };
            dcx.create_err(IntLiteralTooLarge { span, limit })
        }
    }
}

#[derive(Diagnostic)]
#[diag(session_incompatible_linker_flavor)]
#[note]
pub(crate) struct IncompatibleLinkerFlavor {
    pub(crate) flavor: &'static str,
    pub(crate) compatible_list: String,
}

#[derive(Diagnostic)]
#[diag(session_function_return_requires_x86_or_x86_64)]
pub(crate) struct FunctionReturnRequiresX86OrX8664;

#[derive(Diagnostic)]
#[diag(session_function_return_thunk_extern_requires_non_large_code_model)]
pub(crate) struct FunctionReturnThunkExternRequiresNonLargeCodeModel;

#[derive(Diagnostic)]
#[diag(session_indirect_branch_cs_prefix_requires_x86_or_x86_64)]
pub(crate) struct IndirectBranchCsPrefixRequiresX86OrX8664;

#[derive(Diagnostic)]
#[diag(session_unsupported_regparm)]
pub(crate) struct UnsupportedRegparm {
    pub(crate) regparm: u32,
}

#[derive(Diagnostic)]
#[diag(session_unsupported_regparm_arch)]
pub(crate) struct UnsupportedRegparmArch;

#[derive(Diagnostic)]
#[diag(session_unsupported_reg_struct_return_arch)]
pub(crate) struct UnsupportedRegStructReturnArch;

#[derive(Diagnostic)]
#[diag(session_failed_to_create_profiler)]
pub(crate) struct FailedToCreateProfiler {
    pub(crate) err: String,
}

#[derive(Diagnostic)]
#[diag(session_soft_float_ignored)]
#[note]
pub(crate) struct SoftFloatIgnored;

#[derive(Diagnostic)]
#[diag(session_soft_float_deprecated)]
#[note]
#[note(session_soft_float_deprecated_issue)]
pub(crate) struct SoftFloatDeprecated;

#[derive(LintDiagnostic)]
#[diag(session_unexpected_builtin_cfg)]
#[note(session_controlled_by)]
#[note(session_incoherent)]
pub(crate) struct UnexpectedBuiltinCfg {
    pub(crate) cfg: String,
    pub(crate) cfg_name: Symbol,
    pub(crate) controlled_by: &'static str,
}
