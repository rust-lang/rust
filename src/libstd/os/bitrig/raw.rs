// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Bitrig-specific raw type definitions

#![stable(feature = "raw_ext", since = "1.1.0")]
#![rustc_deprecated(since = "1.8.0",
                    reason = "these type aliases are no longer supported by \
                              the standard library, the `libc` crate on \
                              crates.io should be used instead for the correct \
                              definitions")]
#![allow(deprecated)]

use os::raw::c_long;
use os::unix::raw::{uid_t, gid_t};

#[stable(feature = "raw_ext", since = "1.1.0")] pub type blkcnt_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type blksize_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type dev_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type fflags_t = u32;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type ino_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type mode_t = u32;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type nlink_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type off_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type time_t = i64;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = usize;

#[repr(C)]
#[derive(Clone)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub struct stat {
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mode: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_dev: i32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ino: u64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_nlink: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_uid: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_gid: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_rdev: i32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_atime: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_atime_nsec: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mtime: u64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mtime_nsec: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ctime: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ctime_nsec: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_size: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blocks: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blksize: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_flags: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_gen: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_birthtime: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_birthtime_nsec: i64,
}
