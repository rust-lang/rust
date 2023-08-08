use rustc_macros::Diagnostic;
use rustc_session::config::CrateType;
use rustc_span::{Span, Symbol};
use rustc_target::spec::TargetTriple;

use std::io;
use std::path::Path;

#[derive(Diagnostic)]
#[diag(interface_ferris_identifier)]
pub struct FerrisIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "ferris", applicability = "maybe-incorrect")]
    pub first_span: Span,
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
#[diag(interface_cant_emit_mir)]
pub struct CantEmitMIR {
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(interface_rustc_error_fatal)]
pub struct RustcErrorFatal {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(interface_rustc_error_unexpected_annotation)]
pub struct RustcErrorUnexpectedAnnotation {
    #[primary_span]
    pub span: Span,
}

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
#[diag(interface_unsupported_crate_type_for_target)]
pub struct UnsupportedCrateTypeForTarget<'a> {
    pub crate_type: CrateType,
    pub target_triple: &'a TargetTriple,
}

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
