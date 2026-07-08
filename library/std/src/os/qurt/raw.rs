//! QuRT-specific raw type definitions.

#![stable(feature = "raw_ext", since = "1.1.0")]

use core::ffi::c_long;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type ino_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = u32;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type nlink_t = u32;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type off_t = c_long;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type time_t = c_long;

// Threading types for QuRT pthread support
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type pthread_t = libc::pthread_t;
