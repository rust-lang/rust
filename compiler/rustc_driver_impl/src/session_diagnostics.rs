use std::error::Error;

use rustc_macros::{Diagnostic, Subdiagnostic};

#[derive(Diagnostic)]
#[diag("could not emit MIR: {$error}")]
pub struct CantEmitMIR {
    pub error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to read rlink file: `{$err}`")]
pub(crate) struct RlinkUnableToRead {
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("the input does not look like a .rlink file")]
pub(crate) struct RLinkWrongFileType;

#[derive(Diagnostic)]
#[diag("the input does not contain version number")]
pub(crate) struct RLinkEmptyVersionNumber;

#[derive(Diagnostic)]
#[diag(
    ".rlink file was produced with encoding version `{$version_array}`, but the current version is `{$rlink_version}`"
)]
pub(crate) struct RLinkEncodingVersionMismatch {
    pub version_array: String,
    pub rlink_version: u32,
}

#[derive(Diagnostic)]
#[diag(
    ".rlink file was produced by rustc version `{$rustc_version}`, but the current version is `{$current_version}`"
)]
pub(crate) struct RLinkRustcVersionMismatch<'a> {
    pub rustc_version: String,
    pub current_version: &'a str,
}

#[derive(Diagnostic)]
#[diag("rlink must be a file")]
pub(crate) struct RlinkNotAFile;

#[derive(Diagnostic)]
#[diag("corrupt metadata encountered in `{$file}`")]
pub(crate) struct RlinkCorruptFile<'a> {
    pub file: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag("the compiler unexpectedly panicked. this is a bug.")]
pub(crate) struct Ice;

#[derive(Diagnostic)]
#[diag("we would appreciate a bug report: {$bug_report_url}")]
pub(crate) struct IceBugReport<'a> {
    pub bug_report_url: &'a str,
}

#[derive(Diagnostic)]
#[diag("please make sure that you have updated to the latest nightly")]
pub(crate) struct UpdateNightlyNote;

#[derive(Diagnostic)]
#[diag(
    "using internal features is not supported and expected to cause internal compiler errors when used incorrectly"
)]
pub(crate) struct IceBugReportInternalFeature;

#[derive(Diagnostic)]
#[diag("rustc {$version} running on {$triple}")]
pub(crate) struct IceVersion<'a> {
    pub version: &'a str,
    pub triple: &'a str,
}

#[derive(Diagnostic)]
#[diag("please attach the file at `{$path}` to your bug report")]
pub(crate) struct IcePath {
    pub path: std::path::PathBuf,
}

#[derive(Diagnostic)]
#[diag("the ICE couldn't be written to `{$path}`: {$error}")]
pub(crate) struct IcePathError {
    pub path: std::path::PathBuf,
    pub error: String,
    #[subdiagnostic]
    pub env_var: Option<IcePathErrorEnv>,
}

#[derive(Subdiagnostic)]
#[note("the environment variable `RUSTC_ICE` is set to `{$env_var}`")]
pub(crate) struct IcePathErrorEnv {
    pub env_var: std::path::PathBuf,
}

#[derive(Diagnostic)]
#[diag("compiler flags: {$flags}")]
pub(crate) struct IceFlags {
    pub flags: String,
}

#[derive(Diagnostic)]
#[diag("some of the compiler flags provided by cargo are hidden")]
pub(crate) struct IceExcludeCargoDefaults;

#[derive(Diagnostic)]
#[diag("cannot dump feature usage metrics: {$error}")]
pub(crate) struct UnstableFeatureUsage {
    pub error: Box<dyn Error>,
}
