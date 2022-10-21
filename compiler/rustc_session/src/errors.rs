use std::num::NonZeroU32;

use crate::cgu_reuse_tracker::CguReuse;
use rustc_errors::MultiSpan;
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};
use rustc_target::spec::{SplitDebuginfo, StackProtector, TargetTriple};

#[derive(Diagnostic)]
#[diag(session::incorrect_cgu_reuse_type)]
pub struct IncorrectCguReuseType<'a> {
    #[primary_span]
    pub span: Span,
    pub cgu_user_name: &'a str,
    pub actual_reuse: CguReuse,
    pub expected_reuse: CguReuse,
    pub at_least: u8,
}

#[derive(Diagnostic)]
#[diag(session::cgu_not_recorded)]
pub struct CguNotRecorded<'a> {
    pub cgu_user_name: &'a str,
    pub cgu_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(session::feature_gate_error, code = "E0658")]
pub struct FeatureGateError<'a> {
    #[primary_span]
    pub span: MultiSpan,
    pub explain: &'a str,
}

#[derive(Subdiagnostic)]
#[note(session::feature_diagnostic_for_issue)]
pub struct FeatureDiagnosticForIssue {
    pub n: NonZeroU32,
}

#[derive(Subdiagnostic)]
#[help(session::feature_diagnostic_help)]
pub struct FeatureDiagnosticHelp {
    pub feature: Symbol,
}

#[derive(Diagnostic)]
#[diag(session::not_circumvent_feature)]
pub struct NotCircumventFeature;

#[derive(Diagnostic)]
#[diag(session::linker_plugin_lto_windows_not_supported)]
pub struct LinkerPluginToWindowsNotSupported;

#[derive(Diagnostic)]
#[diag(session::profile_use_file_does_not_exist)]
pub struct ProfileUseFileDoesNotExist<'a> {
    pub path: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session::profile_sample_use_file_does_not_exist)]
pub struct ProfileSampleUseFileDoesNotExist<'a> {
    pub path: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session::target_requires_unwind_tables)]
pub struct TargetRequiresUnwindTables;

#[derive(Diagnostic)]
#[diag(session::sanitizer_not_supported)]
pub struct SanitizerNotSupported {
    pub us: String,
}

#[derive(Diagnostic)]
#[diag(session::sanitizers_not_supported)]
pub struct SanitizersNotSupported {
    pub us: String,
}

#[derive(Diagnostic)]
#[diag(session::cannot_mix_and_match_sanitizers)]
pub struct CannotMixAndMatchSanitizers {
    pub first: String,
    pub second: String,
}

#[derive(Diagnostic)]
#[diag(session::cannot_enable_crt_static_linux)]
pub struct CannotEnableCrtStaticLinux;

#[derive(Diagnostic)]
#[diag(session::sanitizer_cfi_enabled)]
pub struct SanitizerCfiEnabled;

#[derive(Diagnostic)]
#[diag(session::unstable_virtual_function_elimination)]
pub struct UnstableVirtualFunctionElimination;

#[derive(Diagnostic)]
#[diag(session::unsupported_dwarf_version)]
pub struct UnsupportedDwarfVersion {
    pub dwarf_version: u32,
}

#[derive(Diagnostic)]
#[diag(session::target_stack_protector_not_supported)]
pub struct StackProtectorNotSupportedForTarget<'a> {
    pub stack_protector: StackProtector,
    pub target_triple: &'a TargetTriple,
}

#[derive(Diagnostic)]
#[diag(session::split_debuginfo_unstable_platform)]
pub struct SplitDebugInfoUnstablePlatform {
    pub debuginfo: SplitDebuginfo,
}

#[derive(Diagnostic)]
#[diag(session::file_is_not_writeable)]
pub struct FileIsNotWriteable<'a> {
    pub file: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag(session::crate_name_does_not_match)]
pub struct CrateNameDoesNotMatch<'a> {
    #[primary_span]
    pub span: Span,
    pub s: &'a str,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(session::crate_name_invalid)]
pub struct CrateNameInvalid<'a> {
    pub s: &'a str,
}

#[derive(Diagnostic)]
#[diag(session::crate_name_empty)]
pub struct CrateNameEmpty {
    #[primary_span]
    pub span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(session::invalid_character_in_create_name)]
pub struct InvalidCharacterInCrateName<'a> {
    #[primary_span]
    pub span: Option<Span>,
    pub character: char,
    pub crate_name: &'a str,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(session::expr_parentheses_needed, applicability = "machine-applicable")]
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
#[diag(session::skipping_const_checks)]
pub struct SkippingConstChecks {
    #[subdiagnostic(eager)]
    pub unleashed_features: Vec<UnleashedFeatureHelp>,
}

#[derive(Subdiagnostic)]
pub enum UnleashedFeatureHelp {
    #[help(session::unleashed_feature_help_named)]
    Named {
        #[primary_span]
        span: Span,
        gate: Symbol,
    },
    #[help(session::unleashed_feature_help_unnamed)]
    Unnamed {
        #[primary_span]
        span: Span,
    },
}
