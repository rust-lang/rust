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
