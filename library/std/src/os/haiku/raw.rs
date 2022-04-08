//! Haiku-specific raw type definitions

#![stable(feature = "raw_ext", since = "1.1.0")]
#![deprecated(
    since = "1.53.0",
    note = "these type aliases are no longer supported by \
            the standard library, the `libc` crate on \
            crates.io should be used instead for the correct \
            definitions"
)]
#![allow(deprecated)]

use crate::os::raw::c_long;
use crate::os::unix::raw::{gid_t, uid_t};

// Use the direct definition of usize, instead of uintptr_t like in libc
#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = usize;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blkcnt_t = i64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blksize_t = i32;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = i32;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type ino_t = i64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = u32;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type nlink_t = i32;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type off_t = i64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type time_t = i32;

#[repr(C)]
#[derive(Clone)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub struct stat {
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_dev: dev_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ino: ino_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mode: mode_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_nlink: nlink_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_uid: uid_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_gid: gid_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_size: off_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_rdev: dev_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blksize: blksize_t,
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
    pub st_crtime: time_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_crtime_nsec: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_type: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blocks: blkcnt_t,
}
