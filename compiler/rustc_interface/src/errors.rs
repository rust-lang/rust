use rustc_macros::Diagnostic;
use rustc_session::config::CrateType;
use rustc_span::{Span, Symbol};
use rustc_target::spec::TargetTriple;

use std::io;
use std::path::Path;

#[derive(Diagnostic)]
#[diag(interface::ferris_identifier)]
pub struct FerrisIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "ferris", applicability = "maybe-incorrect")]
    pub first_span: Span,
}

#[derive(Diagnostic)]
#[diag(interface::emoji_identifier)]
pub struct EmojiIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    pub ident: Symbol,
}

#[derive(Diagnostic)]
#[diag(interface::mixed_bin_crate)]
pub struct MixedBinCrate;

#[derive(Diagnostic)]
#[diag(interface::mixed_proc_macro_crate)]
pub struct MixedProcMacroCrate;

#[derive(Diagnostic)]
#[diag(interface::proc_macro_doc_without_arg)]
pub struct ProcMacroDocWithoutArg;

#[derive(Diagnostic)]
#[diag(interface::error_writing_dependencies)]
pub struct ErrorWritingDependencies<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(interface::input_file_would_be_overwritten)]
pub struct InputFileWouldBeOverWritten<'a> {
    pub path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(interface::generated_file_conflicts_with_directory)]
pub struct GeneratedFileConflictsWithDirectory<'a> {
    pub input_path: &'a Path,
    pub dir_path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(interface::temps_dir_error)]
pub struct TempsDirError;

#[derive(Diagnostic)]
#[diag(interface::out_dir_error)]
pub struct OutDirError;

#[derive(Diagnostic)]
#[diag(interface::cant_emit_mir)]
pub struct CantEmitMIR {
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(interface::rustc_error_fatal)]
pub struct RustcErrorFatal {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(interface::rustc_error_unexpected_annotation)]
pub struct RustcErrorUnexpectedAnnotation {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(interface::failed_writing_file)]
pub struct FailedWritingFile<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(interface::unsupported_crate_type_for_target)]
pub struct UnsupportedCrateTypeForTarget<'a> {
    pub crate_type: CrateType,
    pub target_triple: &'a TargetTriple,
}

#[derive(Diagnostic)]
#[diag(interface::multiple_output_types_adaption)]
pub struct MultipleOutputTypesAdaption;

#[derive(Diagnostic)]
#[diag(interface::ignoring_extra_filename)]
pub struct IgnoringExtraFilename;

#[derive(Diagnostic)]
#[diag(interface::ignoring_out_dir)]
pub struct IgnoringOutDir;
