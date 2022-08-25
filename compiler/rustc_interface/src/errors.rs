use rustc_macros::SessionDiagnostic;
use rustc_span::{Span, Symbol};

use std::io;
use std::path::Path;

#[derive(SessionDiagnostic)]
#[diag(interface::ferris_identifier)]
pub struct FerrisIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "ferris", applicability = "maybe-incorrect")]
    pub first_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(interface::emoji_identifier)]
pub struct EmojiIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    pub ident: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(interface::mixed_bin_crate)]
pub struct MixedBinCrate;

#[derive(SessionDiagnostic)]
#[diag(interface::mixed_proc_macro_crate)]
pub struct MixedProcMacroCrate;

#[derive(SessionDiagnostic)]
#[diag(interface::proc_macro_doc_without_arg)]
pub struct ProcMacroDocWithoutArg;

#[derive(SessionDiagnostic)]
#[diag(interface::error_writing_dependencies)]
pub struct ErrorWritingDependencies<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(SessionDiagnostic)]
#[diag(interface::input_file_would_be_overwritten)]
pub struct InputFileWouldBeOverWritten<'a> {
    pub path: &'a Path,
}

#[derive(SessionDiagnostic)]
#[diag(interface::generated_file_conflicts_with_directory)]
pub struct GeneratedFileConflictsWithDirectory<'a> {
    pub input_path: &'a Path,
    pub dir_path: &'a Path,
}

#[derive(SessionDiagnostic)]
#[diag(interface::temps_dir_error)]
pub struct TempsDirError;

#[derive(SessionDiagnostic)]
#[diag(interface::out_dir_error)]
pub struct OutDirError;

#[derive(SessionDiagnostic)]
#[diag(interface::cant_emit_mir)]
pub struct CantEmitMIR {
    pub error: io::Error,
}

#[derive(SessionDiagnostic)]
#[diag(interface::rustc_error_fatal)]
pub struct RustcErrorFatal {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(interface::rustc_error_unexpected_annotation)]
pub struct RustcErrorUnexpectedAnnotation {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(interface::failed_writing_file)]
pub struct FailedWritingFile<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}
