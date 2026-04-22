#[cfg(test)]
mod tests;

#[cfg_attr(
    test,
    expect(unused, reason = "only used in implementation for non-test compilation")
)]
use alloc_crate::io::OsFunctions;
#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use alloc_crate::io::RawOsError;
#[unstable(feature = "io_const_error_internals", issue = "none")]
pub use alloc_crate::io::SimpleMessage;
#[unstable(feature = "io_const_error", issue = "133448")]
pub use alloc_crate::io::const_error;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::io::{Error, ErrorKind, Result};

#[cfg_attr(
    test,
    expect(unused, reason = "only used in implementation for non-test compilation")
)]
use crate::sys::io::{decode_error_kind, errno, error_string, is_interrupted};

// Because std is linked in during testing, these incoherent implementations would
// be duplicated if this was unconditionally included.
// See #2912 for details.
#[cfg(not(test))]
impl Error {
    /// Returns an error representing the last OS error which occurred.
    ///
    /// This function reads the value of `errno` for the target platform (e.g.
    /// `GetLastError` on Windows) and will return a corresponding instance of
    /// [`Error`] for the error code.
    ///
    /// This should be called immediately after a call to a platform function,
    /// otherwise the state of the error value is indeterminate. In particular,
    /// other standard library functions may call platform functions that may
    /// (or may not) reset the error value even if they succeed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Error;
    ///
    /// let os_error = Error::last_os_error();
    /// println!("last OS error: {os_error:?}");
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[doc(alias = "GetLastError")]
    #[doc(alias = "errno")]
    #[must_use]
    #[inline]
    pub fn last_os_error() -> Error {
        Error::from_raw_os_error(errno())
    }

    /// Creates a new instance of an [`Error`] from a particular OS error code.
    ///
    /// # Examples
    ///
    /// On Linux:
    ///
    /// ```
    /// # if cfg!(target_os = "linux") {
    /// use std::io;
    ///
    /// let error = io::Error::from_raw_os_error(22);
    /// assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    /// # }
    /// ```
    ///
    /// On Windows:
    ///
    /// ```
    /// # if cfg!(windows) {
    /// use std::io;
    ///
    /// let error = io::Error::from_raw_os_error(10022);
    /// assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    /// # }
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub fn from_raw_os_error(code: RawOsError) -> Error {
        const FUNCTIONS: &'static OsFunctions = &OsFunctions {
            format_os_error: |code, fmt| fmt.write_str(&error_string(code)),
            decode_error_kind,
            is_interrupted,
        };

        // SAFETY: `FUNCTIONS` is a constant and not created at runtime.
        unsafe { Error::from_raw_os_error_with_functions(code, FUNCTIONS) }
    }
}
