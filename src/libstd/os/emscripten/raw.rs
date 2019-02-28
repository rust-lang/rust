//! Emscripten-specific raw type definitions
//! This is basically exactly the same as the linux definitions,
//! except using the musl-specific stat64 structure in liblibc.

#![stable(feature = "raw_ext", since = "1.1.0")]
#![rustc_deprecated(since = "1.8.0",
                    reason = "these type aliases are no longer supported by \
                              the standard library, the `libc` crate on \
                              crates.io should be used instead for the correct \
                              definitions")]
#![allow(deprecated)]

use crate::os::raw::{c_long, c_short, c_uint, c_ulong};

#[stable(feature = "raw_ext", since = "1.1.0")] pub type dev_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type mode_t = u32;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = c_ulong;

#[doc(inline)]
#[stable(feature = "raw_ext", since = "1.1.0")] pub type blkcnt_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type blksize_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type ino_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type nlink_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type off_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type time_t = c_long;

#[repr(C)]
#[derive(Clone)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub struct stat {
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_dev: u64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub __pad1: c_short,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub __st_ino: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mode: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_nlink: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_uid: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_gid: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_rdev: u64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub __pad2: c_uint,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_size: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blksize: i32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blocks: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_atime: time_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_atime_nsec: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mtime: time_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mtime_nsec: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ctime: time_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ctime_nsec: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ino: u64,
}
