//! Errors emitted by codegen_ssa

use rustc_macros::SessionDiagnostic;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::missing_native_static_library)]
pub struct MissingNativeStaticLibrary<'a> {
    pub library_name: &'a str,
}
