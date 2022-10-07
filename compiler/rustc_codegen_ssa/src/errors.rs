//! Errors emitted by codegen_ssa

use rustc_macros::SessionDiagnostic;
use std::io::Error;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::missing_native_static_library)]
pub struct MissingNativeStaticLibrary<'a> {
    pub library_name: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::lib_def_write_failure)]
pub struct LibDefWriteFailure {
    pub error: Error,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::version_script_write_failure)]
pub struct VersionScriptWriteFailure {
    pub error: Error,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::symbol_file_write_failure)]
pub struct SymbolFileWriteFailure {
    pub error: Error,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::unsupported_arch)]
pub struct UnsupportedArch;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::msvc_path_not_found)]
pub struct MsvcPathNotFound;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::link_exe_not_found)]
pub struct LinkExeNotFound;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::ld64_unimplemented_modifier)]
pub struct Ld64UnimplementedModifier;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::linker_unsupported_modifier)]
pub struct LinkerUnsupportedModifier;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::L4Bender_exporting_symbols_unimplemented)]
pub struct L4BenderExportingSymbolsUnimplemented;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::no_natvis_directory)]
pub struct NoNatvisDirectory {
    pub error: Error,
}
