use crate::ffi::OsString;

/// Returns the system hostname.
///
/// This can error out in platform-specific error cases;
/// for example, uefi and wasm, where hostnames aren't
/// supported.
#[unstable(feature = "gethostname", issue = "135142")]
pub fn hostname() -> crate::io::Result<OsString> {
    crate::sys_common::net::gethostname()
}
