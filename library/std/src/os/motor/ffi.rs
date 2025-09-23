//! Motor OS-specific extensions to primitives in the [`std::ffi`] module.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::ffi::{OsStr, OsString};
use crate::sealed::Sealed;

/// Motor OS-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt: Sealed {
    /// Motor OS strings are utf-8, and thus just strings.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_str(&self) -> &str;
}

#[stable(feature = "rust1", since = "1.0.0")]
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
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStrExt: Sealed {
    /// Motor OS strings are utf-8, and thus just strings.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_str(&self) -> &str;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStrExt for OsStr {
    #[inline]
    fn as_str(&self) -> &str {
        self.to_str().unwrap()
    }
}
