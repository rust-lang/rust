//! Motor OS-specific extensions to primitives in the [`std::ffi`] module.
#![unstable(feature = "motor_ext", issue = "147456")]

use crate::ffi::{OsStr, OsString};
use crate::sealed::Sealed;
use crate::sys_common::{AsInner, IntoInner};

/// Motor OS–specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
pub trait OsStringExt: Sealed {
    /// Yields the underlying UTF-8 string of this [`OsString`].
    ///
    /// OS strings on Motor OS are guaranteed to be UTF-8, so are just strings.
    fn into_string(self) -> String;
}

impl OsStringExt for OsString {
    #[inline]
    fn into_string(self) -> String {
        self.into_inner().inner
    }
}

/// Motor OS–specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
pub trait OsStrExt: Sealed {
    /// Gets the underlying UTF-8 string view of the [`OsStr`] slice.
    ///
    /// OS strings on Motor OS are guaranteed to be UTF-8, so are just strings.
    fn as_str(&self) -> &str;
}

impl OsStrExt for OsStr {
    #[inline]
    fn as_str(&self) -> &str {
        &self.as_inner().inner
    }
}
