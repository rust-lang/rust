#![stable(feature = "raw_ext", since = "1.1.0")]
#![deprecated(
    since = "1.8.0",
    note = "these type aliases are no longer supported by \
            the standard library, the `libc` crate on \
            crates.io should be used instead for the correct \
            definitions"
)]
#![allow(deprecated)]

use crate::os::raw::c_int;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = u32;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = u32;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = c_int;

#[doc(inline)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub use self::arch::{blkcnt_t, blksize_t, ino_t, nlink_t, off_t, time_t};

mod arch {
    use crate::os::raw::c_long;

    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blkcnt_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blksize_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type ino_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type nlink_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type off_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type time_t = c_long;
}
