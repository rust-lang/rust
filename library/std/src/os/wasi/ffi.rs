//! WASI-specific extensions to primitives in the [`std::ffi`] module
//!
//! [`std::ffi`]: crate::ffi

#![stable(feature = "rust1", since = "1.0.0")]

#[path = "../unix/ffi/os_str.rs"]
mod os_str;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::os_str::{OsStrExt, OsStringExt};

#[cfg(any(all(target_arch = "wasm64", target_os = "wasi")))]
use crate::sys_common::{AsInner, FromInner, IntoInner};

#[cfg(any(all(target_arch = "wasm64", target_os = "wasi")))]
impl crate::ffi::OsStr
{
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_bytes(slice: &[u8]) -> &crate::ffi::OsStr {
        unsafe { crate::mem::transmute(slice) }
    }
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_bytes(&self) -> &[u8] {
        &self.as_inner().inner
    }
}

#[cfg(any(all(target_arch = "wasm64", target_os = "wasi")))]
impl crate::ffi::OsString
{
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_vec(vec: Vec<u8>) -> crate::ffi::OsString {
        FromInner::from_inner(crate::sys::os_str::Buf { inner: vec })
    }
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_vec(self) -> Vec<u8> {
        self.into_inner().inner
    }
}
