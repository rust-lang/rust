//! [`OsStringExt`] for unix.

use super::{private, Buf, OsString};
use crate::vec::Vec;

/// Platform-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt: private::Sealed {
    /// Creates an [`OsString`] from a byte vector.
    ///
    /// See the module documentation for an example.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_vec(vec: Vec<u8>) -> Self;

    /// Yields the underlying byte vector of this [`OsString`].
    ///
    /// See the module documentation for an example.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn into_vec(self) -> Vec<u8>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStringExt for OsString {
    #[inline]
    fn from_vec(vec: Vec<u8>) -> OsString {
        From::from(Buf { inner: vec })
    }
    #[inline]
    fn into_vec(self) -> Vec<u8> {
        self.inner.inner
    }
}
