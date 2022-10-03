//! Errors emitted by codegen_ssa

use crate::back::command::Command;
use rustc_errors::{
    fluent, DiagnosticArgValue, DiagnosticBuilder, ErrorGuaranteed, Handler, IntoDiagnostic,
    IntoDiagnosticArg,
};
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;
use std::io::Error;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

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
        CopyPath { from: DebugArgPath(from), to: DebugArgPath(to), error }
    }
}

struct DebugArgPath<'a>(pub &'a Path);

impl IntoDiagnosticArg for DebugArgPath<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(format!("{:?}", self.0)))
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

#[derive(Diagnostic)]
#[diag(codegen_ssa::create_temp_dir)]
pub struct CreateTempDir {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::incompatible_linking_modifiers)]
pub struct IncompatibleLinkingModifiers;

#[derive(Diagnostic)]
#[diag(codegen_ssa::add_native_library)]
pub struct AddNativeLibrary<'a> {
    pub library_path: &'a str,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::multiple_external_func_decl)]
pub struct MultipleExternalFuncDecl<'a> {
    #[primary_span]
    pub span: Span,
    pub function: Symbol,
    pub library_name: &'a str,
}

pub enum LinkRlibError {
    MissingFormat,
    OnlyRmetaFound { crate_name: Symbol },
    NotFound { crate_name: Symbol },
}

impl IntoDiagnostic<'_, !> for LinkRlibError {
    fn into_diagnostic(self, handler: &Handler) -> DiagnosticBuilder<'_, !> {
        match self {
            LinkRlibError::MissingFormat => {
                handler.struct_fatal(fluent::codegen_ssa::rlib_missing_format)
            }
            LinkRlibError::OnlyRmetaFound { crate_name } => {
                let mut diag = handler.struct_fatal(fluent::codegen_ssa::rlib_only_rmeta_found);
                diag.set_arg("crate_name", crate_name);
                diag
            }
            LinkRlibError::NotFound { crate_name } => {
                let mut diag = handler.struct_fatal(fluent::codegen_ssa::rlib_not_found);
                diag.set_arg("crate_name", crate_name);
                diag
            }
        }
    }
}

#[derive(Diagnostic)]
#[diag(codegen_ssa::thorin_dwarf_linking)]
#[note]
pub struct ThorinDwarfLinking {
    pub thorin_error: ThorinErrorWrapper,
}
pub struct ThorinErrorWrapper(pub thorin::Error);

// FIXME: How should we support translations for external crate errors?
impl IntoDiagnosticArg for ThorinErrorWrapper {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(format!("{:?}", self.0)))
    }
}

pub struct LinkingFailed<'a> {
    pub linker_path: &'a PathBuf,
    pub exit_status: ExitStatus,
    pub command: &'a Command,
    pub escaped_output: &'a str,
}

impl IntoDiagnostic<'_> for LinkingFailed<'_> {
    fn into_diagnostic(self, handler: &Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::codegen_ssa::linking_failed);
        diag.set_arg("linker_path", format!("{}", self.linker_path.display()));
        diag.set_arg("exit_status", format!("{}", self.exit_status));

        diag.note(format!("{:?}", self.command)).note(self.escaped_output);

        // Trying to match an error from OS linkers
        // which by now we have no way to translate.
        if self.escaped_output.contains("undefined reference to") {
            diag.note(fluent::codegen_ssa::extern_funcs_not_found)
                .note(fluent::codegen_ssa::specify_libraries_to_link)
                .note(fluent::codegen_ssa::use_cargo_directive);
        }
        diag
    }
}
