//! Redox-specific raw type definitions

#![stable(feature = "raw_ext", since = "1.1.0")]
#![deprecated(
    since = "1.8.0",
    note = "these type aliases are no longer supported by \
            the standard library, the `libc` crate on \
            crates.io should be used instead for the correct \
            definitions"
)]
#![allow(deprecated)]

use crate::os::raw::{c_char, c_int, c_long, c_ulong, c_void};

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = c_long;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type gid_t = c_int;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = c_int;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type uid_t = c_int;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = *mut c_void;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blkcnt_t = c_ulong;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blksize_t = c_ulong;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type ino_t = c_ulong;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type nlink_t = c_ulong;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type off_t = c_long;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type time_t = c_long;

#[repr(C)]
#[derive(Clone)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub struct stat {
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_dev: dev_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ino: ino_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_nlink: nlink_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mode: mode_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_uid: uid_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_gid: gid_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_rdev: dev_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_size: off_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blksize: blksize_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blocks: blkcnt_t,
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
    pub _pad: [c_char; 24],
}
