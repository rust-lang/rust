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
pub type blksize_t = c_long;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blkcnt_t = c_long;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type time_t = c_long;

// Threading types for QuRT pthread support
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type pthread_t = libc::pthread_t;

/// Matches the layout of `libc::stat` for QuRT.
///
/// QuRT has a minimal stat structure without uid/gid, blksize/blocks,
/// or nanosecond timestamp fields.
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
    pub st_rdev: dev_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_size: off_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_atime: time_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mtime: time_t,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ctime: time_t,
}
