use rustc_macros::DiagnosticHandler;
use rustc_span::{Span, Symbol};

use std::io;
use std::path::Path;

#[derive(DiagnosticHandler)]
#[diag(interface::ferris_identifier)]
pub struct FerrisIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "ferris", applicability = "maybe-incorrect")]
    pub first_span: Span,
}

#[derive(DiagnosticHandler)]
#[diag(interface::emoji_identifier)]
pub struct EmojiIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    pub ident: Symbol,
}

#[derive(DiagnosticHandler)]
#[diag(interface::mixed_bin_crate)]
pub struct MixedBinCrate;

#[derive(DiagnosticHandler)]
#[diag(interface::mixed_proc_macro_crate)]
pub struct MixedProcMacroCrate;

#[derive(DiagnosticHandler)]
#[diag(interface::proc_macro_doc_without_arg)]
pub struct ProcMacroDocWithoutArg;

#[derive(DiagnosticHandler)]
#[diag(interface::error_writing_dependencies)]
pub struct ErrorWritingDependencies<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(DiagnosticHandler)]
#[diag(interface::input_file_would_be_overwritten)]
pub struct InputFileWouldBeOverWritten<'a> {
    pub path: &'a Path,
}

#[derive(DiagnosticHandler)]
#[diag(interface::generated_file_conflicts_with_directory)]
pub struct GeneratedFileConflictsWithDirectory<'a> {
    pub input_path: &'a Path,
    pub dir_path: &'a Path,
}

#[derive(DiagnosticHandler)]
#[diag(interface::temps_dir_error)]
pub struct TempsDirError;

#[derive(DiagnosticHandler)]
#[diag(interface::out_dir_error)]
pub struct OutDirError;

#[derive(DiagnosticHandler)]
#[diag(interface::cant_emit_mir)]
pub struct CantEmitMIR {
    pub error: io::Error,
}

#[derive(DiagnosticHandler)]
#[diag(interface::rustc_error_fatal)]
pub struct RustcErrorFatal {
    #[primary_span]
    pub span: Span,
}

#[derive(DiagnosticHandler)]
#[diag(interface::rustc_error_unexpected_annotation)]
pub struct RustcErrorUnexpectedAnnotation {
    #[primary_span]
    pub span: Span,
}

#[derive(DiagnosticHandler)]
#[diag(interface::failed_writing_file)]
pub struct FailedWritingFile<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}
