use std::io;
use std::path::Path;

use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag(interface_crate_name_does_not_match)]
pub(crate) struct CrateNameDoesNotMatch {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) crate_name: Symbol,
    pub(crate) attr_crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(interface_crate_name_invalid)]
pub(crate) struct CrateNameInvalid<'a> {
    pub(crate) crate_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(interface_ferris_identifier)]
pub struct FerrisIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "{ferris_fix}", applicability = "maybe-incorrect")]
    pub first_span: Span,
    pub ferris_fix: &'static str,
}

#[derive(Diagnostic)]
#[diag(interface_emoji_identifier)]
pub struct EmojiIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    pub ident: Symbol,
}

#[derive(Diagnostic)]
#[diag(interface_mixed_bin_crate)]
pub struct MixedBinCrate;

#[derive(Diagnostic)]
#[diag(interface_mixed_proc_macro_crate)]
pub struct MixedProcMacroCrate;

#[derive(Diagnostic)]
#[diag(interface_error_writing_dependencies)]
pub struct ErrorWritingDependencies<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(interface_input_file_would_be_overwritten)]
pub struct InputFileWouldBeOverWritten<'a> {
    pub path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(interface_generated_file_conflicts_with_directory)]
pub struct GeneratedFileConflictsWithDirectory<'a> {
    pub input_path: &'a Path,
    pub dir_path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(interface_temps_dir_error)]
pub struct TempsDirError;

#[derive(Diagnostic)]
#[diag(interface_out_dir_error)]
pub struct OutDirError;

#[derive(Diagnostic)]
#[diag(interface_failed_writing_file)]
pub struct FailedWritingFile<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(interface_proc_macro_crate_panic_abort)]
pub struct ProcMacroCratePanicAbort;

#[derive(Diagnostic)]
#[diag(interface_multiple_output_types_adaption)]
pub struct MultipleOutputTypesAdaption;

#[derive(Diagnostic)]
#[diag(interface_ignoring_extra_filename)]
pub struct IgnoringExtraFilename;

#[derive(Diagnostic)]
#[diag(interface_ignoring_out_dir)]
pub struct IgnoringOutDir;

#[derive(Diagnostic)]
#[diag(interface_multiple_output_types_to_stdout)]
pub struct MultipleOutputTypesToStdout;

#[derive(Diagnostic)]
#[diag(interface_abi_required_feature)]
#[note]
#[note(interface_abi_required_feature_issue)]
pub(crate) struct AbiRequiredTargetFeature<'a> {
    pub feature: &'a str,
    pub enabled: &'a str,
}

#[derive(Diagnostic)]
#[diag(interface_limit_invalid)]
pub(crate) struct LimitInvalid<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub value_span: Span,
    pub error_str: &'a str,
}
