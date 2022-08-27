use rustc_macros::SessionDiagnostic;

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::ranlib_failure)]
pub(crate) struct RanlibFailure {
    pub exit_code: Option<i32>
}
