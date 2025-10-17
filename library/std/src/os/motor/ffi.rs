//! Motor OS-specific extensions to primitives in the [`std::ffi`] module.
#![unstable(feature = "motor_ext", issue = "147456")]

use crate::ffi::{OsStr, OsString};
use crate::sealed::Sealed;

/// Motor OS-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
pub trait OsStringExt: Sealed {
    /// Yields the underlying UTF-8 string of this [`OsString`].
    fn into_string(self) -> String;
}

impl OsStringExt for OsString {
    #[inline]
    fn into_string(self) -> String {
        // SAFETY: The platform encoding of Motor OS is UTF-8. As
        // from_encoded_bytes_unchecked requires that the input bytes originate
        // from OsStr::as_encoded_bytes and/or valid UTF-8, no OsString can
        // contain invalid UTF-8.
        unsafe { String::from_utf8_unchecked(self.into_encoded_bytes()) }
    }
}

/// Motor OS-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
pub trait OsStrExt: Sealed {
    /// Gets the underlying UTF-8 string view of the [`OsStr`] slice.
    fn as_str(&self) -> &str;
}

impl OsStrExt for OsStr {
    #[inline]
    fn as_str(&self) -> &str {
        // SAFETY: As above.
        unsafe { str::from_utf8_unchecked(self.as_encoded_bytes()) }
    }
}
