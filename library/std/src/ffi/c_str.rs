//! [`CStr`], [`CString`], and related types.

#[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
pub use alloc::ffi::c_str::FromVecWithNulError;
#[stable(feature = "cstring_into", since = "1.7.0")]
pub use alloc::ffi::c_str::IntoStringError;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc::ffi::c_str::{CString, NulError};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::ffi::c_str::CStr;
#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
pub use core::ffi::c_str::FromBytesUntilNulError;
#[stable(feature = "cstr_from_bytes", since = "1.10.0")]
pub use core::ffi::c_str::FromBytesWithNulError;
