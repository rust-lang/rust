use rustc_codegen_ssa::session_diagnostic::DeserializeRlinkError;
use rustc_macros::SessionDiagnostic;

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_unable_to_read)]
pub(crate) struct RlinkUnableToRead {
    pub err: std::io::Error,
}

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_unable_to_deserialize)]
pub(crate) struct RlinkUnableToDeserialize {
    pub err: DeserializeRlinkError,
}

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_no_a_file)]
pub(crate) struct RlinkNotAFile;
