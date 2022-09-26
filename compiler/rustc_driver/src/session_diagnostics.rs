use rustc_macros::Diagnostic;

#[derive(Diagnostic)]
#[diag(driver::rlink_unable_to_read)]
pub(crate) struct RlinkUnableToRead {
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(driver::rlink_wrong_file_type)]
pub(crate) struct RLinkWrongFileType;

#[derive(Diagnostic)]
#[diag(driver::rlink_empty_version_number)]
pub(crate) struct RLinkEmptyVersionNumber;

#[derive(Diagnostic)]
#[diag(driver::rlink_encoding_version_mismatch)]
pub(crate) struct RLinkEncodingVersionMismatch {
    pub version_array: String,
    pub rlink_version: u32,
}

#[derive(Diagnostic)]
#[diag(driver::rlink_rustc_version_mismatch)]
pub(crate) struct RLinkRustcVersionMismatch<'a> {
    pub rustc_version: String,
    pub current_version: &'a str,
}

#[derive(Diagnostic)]
#[diag(driver::rlink_no_a_file)]
pub(crate) struct RlinkNotAFile;

#[derive(Diagnostic)]
#[diag(driver::unpretty_dump_fail)]
pub(crate) struct UnprettyDumpFail {
    pub path: String,
    pub err: String,
}
