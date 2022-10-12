use std::num::NonZeroU32;

use crate::cgu_reuse_tracker::CguReuse;
use rustc_errors::{
    fluent, DiagnosticBuilder, ErrorGuaranteed, Handler, IntoDiagnostic, MultiSpan,
};
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};
use rustc_target::abi::TargetDataLayoutErrors;
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

pub struct TargetDataLayoutErrorsWrapper<'a>(pub TargetDataLayoutErrors<'a>);

impl IntoDiagnostic<'_, !> for TargetDataLayoutErrorsWrapper<'_> {
    fn into_diagnostic(self, handler: &Handler) -> DiagnosticBuilder<'_, !> {
        let mut diag;
        match self.0 {
            TargetDataLayoutErrors::InvalidAddressSpace { addr_space, err, cause } => {
                diag = handler.struct_fatal(fluent::session::target_invalid_address_space);
                diag.set_arg("addr_space", addr_space);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::InvalidBits { kind, bit, cause, err } => {
                diag = handler.struct_fatal(fluent::session::target_invalid_bits);
                diag.set_arg("kind", kind);
                diag.set_arg("bit", bit);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::MissingAlignment { cause } => {
                diag = handler.struct_fatal(fluent::session::target_missing_alignment);
                diag.set_arg("cause", cause);
                diag
            }
            TargetDataLayoutErrors::InvalidAlignment { cause, err } => {
                diag = handler.struct_fatal(fluent::session::target_invalid_alignment);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::InconsistentTargetArchitecture { dl, target } => {
                diag = handler.struct_fatal(fluent::session::target_inconsistent_architecture);
                diag.set_arg("dl", dl);
                diag.set_arg("target", target);
                diag
            }
            TargetDataLayoutErrors::InconsistentTargetPointerWidth { pointer_size, target } => {
                diag = handler.struct_fatal(fluent::session::target_inconsistent_pointer_width);
                diag.set_arg("pointer_size", pointer_size);
                diag.set_arg("target", target);
                diag
            }
            TargetDataLayoutErrors::InvalidBitsSize { err } => {
                diag = handler.struct_fatal(fluent::session::target_invalid_bits_size);
                diag.set_arg("err", err);
                diag
            }
        }
    }
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

pub struct InvalidCharacterInCrateName<'a> {
    pub span: Option<Span>,
    pub character: char,
    pub crate_name: &'a str,
}

impl IntoDiagnostic<'_> for InvalidCharacterInCrateName<'_> {
    fn into_diagnostic(self, sess: &Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.struct_err(fluent::session::invalid_character_in_create_name);
        if let Some(sp) = self.span {
            diag.set_span(sp);
        }
        diag.set_arg("character", self.character);
        diag.set_arg("crate_name", self.crate_name);
        diag
    }
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
