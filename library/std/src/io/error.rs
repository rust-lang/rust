//! OS-dependent public methods of `Error`

#[cfg(not(test))]
use alloc::io::{Error, RawOsError};

const _: () = {
    let _ = crate::sys::io::error_string;
    let _ = crate::sys::io::decode_error_kind;
    let _ = crate::sys::io::is_interrupted;
};

// FIXME: for some reason, enabling these in `test` config makes them defined twice,
// but it does not seem to be the case with other incoherent items.
#[cfg(not(test))]
impl Error {
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
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn from_raw_os_error(code: RawOsError) -> Error {
        Error::_from_raw_os_error(
            code,
            &alloc::io::OsFunctions {
                error_string: crate::sys::io::error_string,
                decode_error_kind: crate::sys::io::decode_error_kind,
                is_interrupted: crate::sys::io::is_interrupted,
            },
        )
    }

    /// Returns the OS error that this error represents (if any).
    ///
    /// If this [`Error`] was constructed via `last_os_error` or
    /// `from_raw_os_error`, then this function will return [`Some`], otherwise
    /// it will return [`None`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_os_error(err: &Error) {
    ///     if let Some(raw_os_err) = err.raw_os_error() {
    ///         println!("raw OS error: {raw_os_err:?}");
    ///     } else {
    ///         println!("Not an OS error");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Will print "raw OS error: ...".
    ///     print_os_error(&Error::last_os_error());
    ///     // Will print "Not an OS error".
    ///     print_os_error(&Error::new(ErrorKind::Other, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn raw_os_error(&self) -> Option<RawOsError> {
        self._raw_os_error()
    }

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
    #[stable(feature = "rust1", since = "1.0.0")]
    #[doc(alias = "GetLastError")]
    #[doc(alias = "errno")]
    #[must_use]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn last_os_error() -> Error {
        Error::from_raw_os_error(crate::sys::io::errno())
    }
}
