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

use os::raw::c_long;
use os::unix::raw::{uid_t, gid_t};

pub type blkcnt_t = i64;
pub type blksize_t = u32;
pub type dev_t = i32;
pub type fflags_t = u32; // type not declared, but struct stat have u_int32_t
pub type ino_t = u64;
pub type mode_t = u32;
pub type nlink_t = u32;
pub type off_t = i64;
pub type time_t = i64;

#[repr(C)]
pub struct stat {
    pub st_mode: mode_t,
    pub st_dev: dev_t,
    pub st_ino: ino_t,
    pub st_nlink: nlink_t,
    pub st_uid: uid_t,
    pub st_gid: gid_t,
    pub st_rdev: dev_t,
    pub st_atime: time_t,
    pub st_atime_nsec: c_long,
    pub st_mtime: time_t,
    pub st_mtime_nsec: c_long,
    pub st_ctime: time_t,
    pub st_ctime_nsec: c_long,
    pub st_size: off_t,
    pub st_blocks: blkcnt_t,
    pub st_blksize: blksize_t,
    pub st_flags: fflags_t,
    pub st_gen: u32,
    pub st_birthtime: time_t,
    pub st_birthtime_nsec: c_long,
}
