use std::num::NonZeroU32;

use rustc_errors::{error_code, DiagnosticMessage, ErrorGuaranteed, IntoDiagnostic, MultiSpan};
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};
use rustc_target::spec::{SplitDebuginfo, StackProtector, TargetTriple};

pub struct FeatureGateError {
    pub span: MultiSpan,
    pub explain: DiagnosticMessage,
}

impl<'a> IntoDiagnostic<'a> for FeatureGateError {
    #[track_caller]
    fn into_diagnostic(
        self,
        dcx: &'a rustc_errors::DiagCtxt,
    ) -> rustc_errors::DiagnosticBuilder<'a, ErrorGuaranteed> {
        let mut diag = dcx.struct_err(self.explain);
        diag.set_span(self.span);
        diag.code(error_code!(E0658));
        diag
    }
}

#[derive(Subdiagnostic)]
#[note(session_feature_diagnostic_for_issue)]
pub struct FeatureDiagnosticForIssue {
    pub n: NonZeroU32,
}

#[derive(Subdiagnostic)]
#[help(session_feature_diagnostic_help)]
pub struct FeatureDiagnosticHelp {
    pub feature: Symbol,
}

#[derive(Subdiagnostic)]
#[help(session_cli_feature_diagnostic_help)]
pub struct CliFeatureDiagnosticHelp {
    pub feature: Symbol,
}

#[derive(Diagnostic)]
#[diag(session_not_circumvent_feature)]
pub struct NotCircumventFeature;

#[derive(Diagnostic)]
#[diag(session_linker_plugin_lto_windows_not_supported)]
pub struct LinkerPluginToWindowsNotSupported;

#[derive(Diagnostic)]
#[diag(session_profile_use_file_does_not_exist)]
pub struct ProfileUseFileDoesNotExist<'a> {
    pub path: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session_profile_sample_use_file_does_not_exist)]
pub struct ProfileSampleUseFileDoesNotExist<'a> {
    pub path: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session_target_requires_unwind_tables)]
pub struct TargetRequiresUnwindTables;

#[derive(Diagnostic)]
#[diag(session_instrumentation_not_supported)]
pub struct InstrumentationNotSupported {
    pub us: String,
}

#[derive(Diagnostic)]
#[diag(session_sanitizer_not_supported)]
pub struct SanitizerNotSupported {
    pub us: String,
}

#[derive(Diagnostic)]
#[diag(session_sanitizers_not_supported)]
pub struct SanitizersNotSupported {
    pub us: String,
}

#[derive(Diagnostic)]
#[diag(session_cannot_mix_and_match_sanitizers)]
pub struct CannotMixAndMatchSanitizers {
    pub first: String,
    pub second: String,
}

#[derive(Diagnostic)]
#[diag(session_cannot_enable_crt_static_linux)]
pub struct CannotEnableCrtStaticLinux;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_requires_lto)]
pub struct SanitizerCfiRequiresLto;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_requires_single_codegen_unit)]
pub struct SanitizerCfiRequiresSingleCodegenUnit;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_canonical_jump_tables_requires_cfi)]
pub struct SanitizerCfiCanonicalJumpTablesRequiresCfi;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_generalize_pointers_requires_cfi)]
pub struct SanitizerCfiGeneralizePointersRequiresCfi;

#[derive(Diagnostic)]
#[diag(session_sanitizer_cfi_normalize_integers_requires_cfi)]
pub struct SanitizerCfiNormalizeIntegersRequiresCfi;

#[derive(Diagnostic)]
#[diag(session_split_lto_unit_requires_lto)]
pub struct SplitLtoUnitRequiresLto;

#[derive(Diagnostic)]
#[diag(session_unstable_virtual_function_elimination)]
pub struct UnstableVirtualFunctionElimination;

#[derive(Diagnostic)]
#[diag(session_unsupported_dwarf_version)]
pub struct UnsupportedDwarfVersion {
    pub dwarf_version: u32,
}

#[derive(Diagnostic)]
#[diag(session_target_stack_protector_not_supported)]
pub struct StackProtectorNotSupportedForTarget<'a> {
    pub stack_protector: StackProtector,
    pub target_triple: &'a TargetTriple,
}

#[derive(Diagnostic)]
#[diag(session_branch_protection_requires_aarch64)]
pub(crate) struct BranchProtectionRequiresAArch64;

#[derive(Diagnostic)]
#[diag(session_split_debuginfo_unstable_platform)]
pub struct SplitDebugInfoUnstablePlatform {
    pub debuginfo: SplitDebuginfo,
}

#[derive(Diagnostic)]
#[diag(session_file_is_not_writeable)]
pub struct FileIsNotWriteable<'a> {
    pub file: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session_file_write_fail)]
pub(crate) struct FileWriteFail<'a> {
    pub path: &'a std::path::Path,
    pub err: String,
}

#[derive(Diagnostic)]
#[diag(session_crate_name_does_not_match)]
pub struct CrateNameDoesNotMatch {
    #[primary_span]
    pub span: Span,
    pub s: Symbol,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(session_crate_name_invalid)]
pub struct CrateNameInvalid<'a> {
    pub s: &'a str,
}

#[derive(Diagnostic)]
#[diag(session_crate_name_empty)]
pub struct CrateNameEmpty {
    #[primary_span]
    pub span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(session_invalid_character_in_create_name)]
pub struct InvalidCharacterInCrateName {
    #[primary_span]
    pub span: Option<Span>,
    pub character: char,
    pub crate_name: Symbol,
    #[subdiagnostic]
    pub crate_name_help: Option<InvalidCrateNameHelp>,
}

#[derive(Subdiagnostic)]
pub enum InvalidCrateNameHelp {
    #[help(session_invalid_character_in_create_name_help)]
    AddCrateName,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(session_expr_parentheses_needed, applicability = "machine-applicable")]
pub struct ExprParenthesesNeeded {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

impl ExprParenthesesNeeded {
    pub fn surrounding(s: Span) -> Self {
        ExprParenthesesNeeded { left: s.shrink_to_lo(), right: s.shrink_to_hi() }
    }
}

#[derive(Diagnostic)]
#[diag(session_skipping_const_checks)]
pub struct SkippingConstChecks {
    #[subdiagnostic]
    pub unleashed_features: Vec<UnleashedFeatureHelp>,
}

#[derive(Subdiagnostic)]
pub enum UnleashedFeatureHelp {
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
#[diag(session_optimization_fuel_exhausted)]
pub struct OptimisationFuelExhausted {
    pub msg: String,
}

#[derive(Diagnostic)]
#[diag(session_incompatible_linker_flavor)]
#[note]
pub struct IncompatibleLinkerFlavor {
    pub flavor: &'static str,
    pub compatible_list: String,
}

#[derive(Diagnostic)]
#[diag(session_function_return_requires_x86_or_x86_64)]
pub(crate) struct FunctionReturnRequiresX86OrX8664;

#[derive(Diagnostic)]
#[diag(session_function_return_thunk_extern_requires_non_large_code_model)]
pub(crate) struct FunctionReturnThunkExternRequiresNonLargeCodeModel;

#[derive(Diagnostic)]
#[diag(session_failed_to_create_profiler)]
pub struct FailedToCreateProfiler {
    pub err: String,
}
