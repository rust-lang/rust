//! Errors emitted by codegen_ssa

use rustc_errors::{DiagnosticArgValue, IntoDiagnosticArg};
use rustc_macros::Diagnostic;
use std::borrow::Cow;
use std::io::Error;
use std::path::{Path, PathBuf};

#[derive(Diagnostic)]
#[diag(codegen_ssa::lib_def_write_failure)]
pub struct LibDefWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::version_script_write_failure)]
pub struct VersionScriptWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::symbol_file_write_failure)]
pub struct SymbolFileWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::unsupported_arch)]
pub struct UnsupportedArch;

#[derive(Diagnostic)]
#[diag(codegen_ssa::msvc_path_not_found)]
pub struct MsvcPathNotFound;

#[derive(Diagnostic)]
#[diag(codegen_ssa::link_exe_not_found)]
pub struct LinkExeNotFound;

#[derive(Diagnostic)]
#[diag(codegen_ssa::ld64_unimplemented_modifier)]
pub struct Ld64UnimplementedModifier;

#[derive(Diagnostic)]
#[diag(codegen_ssa::linker_unsupported_modifier)]
pub struct LinkerUnsupportedModifier;

#[derive(Diagnostic)]
#[diag(codegen_ssa::L4Bender_exporting_symbols_unimplemented)]
pub struct L4BenderExportingSymbolsUnimplemented;

#[derive(Diagnostic)]
#[diag(codegen_ssa::no_natvis_directory)]
pub struct NoNatvisDirectory {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::copy_path_buf)]
pub struct CopyPathBuf {
    pub source_file: PathBuf,
    pub output_path: PathBuf,
    pub error: Error,
}

// Reports Paths using `Debug` implementation rather than Path's `Display` implementation.
#[derive(Diagnostic)]
#[diag(codegen_ssa::copy_path)]
pub struct CopyPath<'a> {
    from: DebugArgPath<'a>,
    to: DebugArgPath<'a>,
    error: Error,
}

impl<'a> CopyPath<'a> {
    pub fn new(from: &'a Path, to: &'a Path, error: Error) -> CopyPath<'a> {
        CopyPath { from: DebugArgPath { path: from }, to: DebugArgPath { path: to }, error }
    }
}

struct DebugArgPath<'a> {
    pub path: &'a Path,
}

impl IntoDiagnosticArg for DebugArgPath<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(format!("{:?}", self.path)))
    }
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::ignoring_emit_path)]
pub struct IgnoringEmitPath {
    pub extension: String,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::ignoring_output)]
pub struct IgnoringOutput {
    pub extension: String,
}
