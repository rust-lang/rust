use rustc_macros::SessionDiagnostic;

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_unable_to_read)]
pub(crate) struct RlinkUnableToRead {
    pub error_message: String,
}

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_unable_to_deserialize)]
pub(crate) struct RlinkUnableToDeserialize {
    pub error_message: String,
}

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_no_a_file)]
pub(crate) struct RlinkNotAFile {}
