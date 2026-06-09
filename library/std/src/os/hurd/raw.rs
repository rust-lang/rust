//! Hurd-specific raw type definitions

#![stable(feature = "raw_ext", since = "1.1.0")]
#![deprecated(
    since = "1.8.0",
    note = "these type aliases are no longer supported by \
              the standard library, the `libc` crate on \
              crates.io should be used instead for the correct \
              definitions"
)]
#![allow(deprecated)]

use crate::os::raw::{c_long, c_uint, c_ulong};

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blkcnt_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blksize_t = c_long;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = c_ulong;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type ino_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = c_uint;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type nlink_t = c_ulong;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type off_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type time_t = c_long;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = c_long;
