//! Motor OS-specific extensions to primitives in the [`std::ffi`] module.
#![unstable(feature = "motor_ext", issue = "147456")]

use crate::ffi::{OsStr, OsString};
use crate::sealed::Sealed;

/// Motor OS-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
pub trait OsStringExt: Sealed {
    /// Motor OS strings are utf-8, and thus just strings.
    fn as_str(&self) -> &str;
}

impl OsStringExt for OsString {
    #[inline]
    fn as_str(&self) -> &str {
        self.to_str().unwrap()
    }
}

/// Motor OS-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
pub trait OsStrExt: Sealed {
    /// Motor OS strings are utf-8, and thus just strings.
    fn as_str(&self) -> &str;
}

impl OsStrExt for OsStr {
    #[inline]
    fn as_str(&self) -> &str {
        self.to_str().unwrap()
    }
}
