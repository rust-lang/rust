//! QuRT-specific extension to the primitives in the [`std::ffi`] module.
//!
//! [`std::ffi`]: crate::ffi

#![stable(feature = "raw_ext", since = "1.1.0")]

use crate::ffi::{OsStr, OsString};
use crate::mem;
use crate::sealed::Sealed;
use crate::sys::os_str::Buf;
use crate::sys::{AsInner, FromInner, IntoInner};

/// QuRT-specific extensions to [`OsString`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
#[stable(feature = "raw_ext", since = "1.1.0")]
pub trait OsStringExt: Sealed {
    /// Creates an [`OsString`] from a byte vector.
    #[stable(feature = "raw_ext", since = "1.1.0")]
    fn from_vec(vec: Vec<u8>) -> Self;

    /// Yields the underlying byte vector of this [`OsString`].
    #[stable(feature = "raw_ext", since = "1.1.0")]
    fn into_vec(self) -> Vec<u8>;
}

#[stable(feature = "raw_ext", since = "1.1.0")]
impl OsStringExt for OsString {
    #[inline]
    fn from_vec(vec: Vec<u8>) -> OsString {
        FromInner::from_inner(Buf { inner: vec })
    }

    #[inline]
    fn into_vec(self) -> Vec<u8> {
        self.into_inner().inner
    }
}

/// QuRT-specific extensions to [`OsStr`].
///
/// This trait is sealed: it cannot be implemented outside the standard library.
#[stable(feature = "raw_ext", since = "1.1.0")]
pub trait OsStrExt: Sealed {
    #[stable(feature = "raw_ext", since = "1.1.0")]
    /// Creates an [`OsStr`] from a byte slice.
    fn from_bytes(slice: &[u8]) -> &Self;

    /// Gets the underlying byte view of the [`OsStr`] slice.
    #[stable(feature = "raw_ext", since = "1.1.0")]
    fn as_bytes(&self) -> &[u8];
}

#[stable(feature = "raw_ext", since = "1.1.0")]
impl OsStrExt for OsStr {
    #[inline]
    fn from_bytes(slice: &[u8]) -> &OsStr {
        unsafe { mem::transmute(slice) }
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.as_inner().inner
    }
}
