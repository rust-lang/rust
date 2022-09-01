use std::num::NonZeroU32;

use crate::cgu_reuse_tracker::CguReuse;
use crate::{self as rustc_session, SessionDiagnostic};
use rustc_errors::{fluent, DiagnosticBuilder, ErrorGuaranteed, Handler, MultiSpan};
use rustc_macros::SessionDiagnostic;
use rustc_span::{Span, Symbol};
use rustc_target::abi::TargetDataLayoutErrors;
use rustc_target::spec::{SplitDebuginfo, StackProtector, TargetTriple};

#[derive(SessionDiagnostic)]
#[diag(session::incorrect_cgu_reuse_type)]
pub struct IncorrectCguReuseType<'a> {
    #[primary_span]
    pub span: Span,
    pub cgu_user_name: &'a str,
    pub actual_reuse: CguReuse,
    pub expected_reuse: CguReuse,
    pub at_least: u8,
}

#[derive(SessionDiagnostic)]
#[diag(session::cgu_not_recorded)]
pub struct CguNotRecorded<'a> {
    pub cgu_user_name: &'a str,
    pub cgu_name: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(session::feature_gate_error, code = "E0658")]
pub struct FeatureGateError<'a> {
    #[primary_span]
    pub span: MultiSpan,
    pub explain: &'a str,
}

#[derive(SessionSubdiagnostic)]
#[note(session::feature_diagnostic_for_issue)]
pub struct FeatureDiagnosticForIssue {
    pub n: NonZeroU32,
}

#[derive(SessionSubdiagnostic)]
#[help(session::feature_diagnostic_help)]
pub struct FeatureDiagnosticHelp {
    pub feature: Symbol,
}

impl SessionDiagnostic<'_, !> for TargetDataLayoutErrors<'_> {
    fn into_diagnostic(self, sess: &Handler) -> DiagnosticBuilder<'_, !> {
        let mut diag;
        match self {
            TargetDataLayoutErrors::InvalidAddressSpace { addr_space, err, cause } => {
                diag = sess.struct_fatal(fluent::session::target_invalid_address_space);
                diag.set_arg("addr_space", addr_space);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::InvalidBits { kind, bit, cause, err } => {
                diag = sess.struct_fatal(fluent::session::target_invalid_bits);
                diag.set_arg("kind", kind);
                diag.set_arg("bit", bit);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::MissingAlignment { cause } => {
                diag = sess.struct_fatal(fluent::session::target_missing_alignment);
                diag.set_arg("cause", cause);
                diag
            }
            TargetDataLayoutErrors::InvalidAlignment { cause, err } => {
                diag = sess.struct_fatal(fluent::session::target_invalid_alignment);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::InconsistentTargetArchitecture { dl, target } => {
                diag = sess.struct_fatal(fluent::session::target_inconsistent_architecture);
                diag.set_arg("dl", dl);
                diag.set_arg("target", target);
                diag
            }
            TargetDataLayoutErrors::InconsistentTargetPointerWidth { pointer_size, target } => {
                diag = sess.struct_fatal(fluent::session::target_inconsistent_pointer_width);
                diag.set_arg("pointer_size", pointer_size);
                diag.set_arg("target", target);
                diag
            }
            TargetDataLayoutErrors::InvalidBitsSize { err } => {
                diag = sess.struct_fatal(fluent::session::target_invalid_bits_size);
                diag.set_arg("err", err);
                diag
            }
        }
    }
}

#[derive(SessionDiagnostic)]
#[diag(session::not_circumvent_feature)]
pub struct NotCircumventFeature;

#[derive(SessionDiagnostic)]
#[diag(session::linker_plugin_lto_windows_not_supported)]
pub struct LinkerPluginToWindowsNotSupported;

#[derive(SessionDiagnostic)]
#[diag(session::profile_use_file_does_not_exist)]
pub struct ProfileUseFileDoesNotExist<'a> {
    pub path: &'a std::path::Path,
}

#[derive(SessionDiagnostic)]
#[diag(session::profile_sample_use_file_does_not_exist)]
pub struct ProfileSampleUseFileDoesNotExist<'a> {
    pub path: &'a std::path::Path,
}

#[derive(SessionDiagnostic)]
#[diag(session::target_requires_unwind_tables)]
pub struct TargetRequiresUnwindTables;

#[derive(SessionDiagnostic)]
#[diag(session::sanitizer_not_supported)]
pub struct SanitizerNotSupported {
    pub us: String,
}

#[derive(SessionDiagnostic)]
#[diag(session::sanitizers_not_supported)]
pub struct SanitizersNotSupported {
    pub us: String,
}

#[derive(SessionDiagnostic)]
#[diag(session::cannot_mix_and_match_sanitizers)]
pub struct CannotMixAndMatchSanitizers {
    pub first: String,
    pub second: String,
}

#[derive(SessionDiagnostic)]
#[diag(session::cannot_enable_crt_static_linux)]
pub struct CannotEnableCrtStaticLinux;

#[derive(SessionDiagnostic)]
#[diag(session::sanitizer_cfi_enabled)]
pub struct SanitizerCfiEnabled;

#[derive(SessionDiagnostic)]
#[diag(session::unstable_virtual_function_elimination)]
pub struct UnstableVirtualFunctionElimination;

#[derive(SessionDiagnostic)]
#[diag(session::unsupported_dwarf_version)]
pub struct UnsupportedDwarfVersion {
    pub dwarf_version: u32,
}

#[derive(SessionDiagnostic)]
#[diag(session::target_stack_protector_not_supported)]
pub struct StackProtectorNotSupportedForTarget<'a> {
    pub stack_protector: StackProtector,
    pub target_triple: &'a TargetTriple,
}

#[derive(SessionDiagnostic)]
#[diag(session::split_debuginfo_unstable_platform)]
pub struct SplitDebugInfoUnstablePlatform {
    pub debuginfo: SplitDebuginfo,
}

#[derive(SessionDiagnostic)]
#[diag(session::file_is_not_writeable)]
pub struct FileIsNotWriteable<'a> {
    pub file: &'a std::path::Path,
}

#[derive(SessionDiagnostic)]
#[diag(session::crate_name_does_not_match)]
pub struct CrateNameDoesNotMatch<'a> {
    #[primary_span]
    pub span: Span,
    pub s: &'a str,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(session::crate_name_invalid)]
pub struct CrateNameInvalid<'a> {
    pub s: &'a str,
}

#[derive(SessionDiagnostic)]
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

impl crate::SessionDiagnostic<'_> for InvalidCharacterInCrateName<'_> {
    fn into_diagnostic(
        self,
        sess: &Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.struct_err(fluent::session::invalid_character_in_create_name);
        if let Some(sp) = self.span {
            diag.set_span(sp);
        }
        diag.set_arg("character", self.character);
        diag.set_arg("crate_name", self.crate_name);
        diag
    }
}
