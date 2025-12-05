use crate::ffi::OsString;

/// Returns the system hostname.
///
/// This can error out in platform-specific error cases;
/// for example, uefi and wasm, where hostnames aren't
/// supported.
///
/// # Underlying system calls
///
/// | Platform | System call                                                                                             |
/// |----------|---------------------------------------------------------------------------------------------------------|
/// | UNIX     | [`gethostname`](https://www.man7.org/linux/man-pages/man2/gethostname.2.html)                           |
/// | Windows  | [`GetHostNameW`](https://learn.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-gethostnamew) |
///
/// Note that platform-specific behavior [may change in the future][changes].
///
/// [changes]: crate::io#platform-specific-behavior
#[unstable(feature = "gethostname", issue = "135142")]
pub fn hostname() -> crate::io::Result<OsString> {
    crate::sys::net::hostname()
}
