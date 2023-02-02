use rustc_macros::Diagnostic;

#[derive(Diagnostic)]
#[diag(driver_rlink_unable_to_read)]
pub(crate) struct RlinkUnableToRead {
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(driver_rlink_wrong_file_type)]
pub(crate) struct RLinkWrongFileType;

#[derive(Diagnostic)]
#[diag(driver_rlink_empty_version_number)]
pub(crate) struct RLinkEmptyVersionNumber;

#[derive(Diagnostic)]
#[diag(driver_rlink_encoding_version_mismatch)]
pub(crate) struct RLinkEncodingVersionMismatch {
    pub version_array: String,
    pub rlink_version: u32,
}

#[derive(Diagnostic)]
#[diag(driver_rlink_rustc_version_mismatch)]
pub(crate) struct RLinkRustcVersionMismatch<'a> {
    pub rustc_version: String,
    pub current_version: &'a str,
}

#[derive(Diagnostic)]
#[diag(driver_rlink_no_a_file)]
pub(crate) struct RlinkNotAFile;

#[derive(Diagnostic)]
#[diag(driver_unpretty_dump_fail)]
pub(crate) struct UnprettyDumpFail {
    pub path: String,
    pub err: String,
}

#[derive(Diagnostic)]
#[diag(driver_ice)]
pub(crate) struct Ice;

#[derive(Diagnostic)]
#[diag(driver_ice_bug_report)]
pub(crate) struct IceBugReport<'a> {
    pub bug_report_url: &'a str,
}

#[derive(Diagnostic)]
#[diag(driver_ice_version)]
pub(crate) struct IceVersion<'a> {
    pub version: &'a str,
    pub triple: &'a str,
}

#[derive(Diagnostic)]
#[diag(driver_ice_flags)]
pub(crate) struct IceFlags {
    pub flags: String,
}

#[derive(Diagnostic)]
#[diag(driver_ice_exclude_cargo_defaults)]
pub(crate) struct IceExcludeCargoDefaults;
