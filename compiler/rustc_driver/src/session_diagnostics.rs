use rustc_macros::SessionDiagnostic;

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_unable_to_read)]
pub(crate) struct RlinkUnableToRead {
    pub err: std::io::Error,
}

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_wrong_file_type)]
pub(crate) struct RLinkWrongFileType;

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_empty_version_number)]
pub(crate) struct RLinkEmptyVersionNumber;

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_encoding_version_mismatch)]
pub(crate) struct RLinkEncodingVersionMismatch {
    pub version_array: String,
    pub rlink_version: String,
}

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_rustc_version_mismatch)]
pub(crate) struct RLinkRustcVersionMismatch {
    pub rustc_version: String,
    pub current_version: String,
}

#[derive(SessionDiagnostic)]
#[diag(driver::rlink_no_a_file)]
pub(crate) struct RlinkNotAFile;
