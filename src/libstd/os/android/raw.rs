// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Android-specific raw type definitions

use os::raw::{c_uint, c_uchar, c_ulonglong, c_longlong, c_ulong};
use os::unix::raw::{uid_t, gid_t};

pub type blkcnt_t = u32;
pub type blksize_t = u32;
pub type dev_t = u32;
pub type ino_t = u32;
pub type mode_t = u16;
pub type nlink_t = u16;
pub type off_t = i32;
pub type time_t = i32;

#[repr(C)]
pub struct stat {
    pub st_dev: c_ulonglong,
    pub __pad0: [c_uchar; 4],
    pub __st_ino: ino_t,
    pub st_mode: c_uint,
    pub st_nlink: c_uint,
    pub st_uid: uid_t,
    pub st_gid: gid_t,
    pub st_rdev: c_ulonglong,
    pub __pad3: [c_uchar; 4],
    pub st_size: c_longlong,
    pub st_blksize: blksize_t,
    pub st_blocks: c_ulonglong,
    pub st_atime: time_t,
    pub st_atime_nsec: c_ulong,
    pub st_mtime: time_t,
    pub st_mtime_nsec: c_ulong,
    pub st_ctime: time_t,
    pub st_ctime_nsec: c_ulong,
    pub st_ino: c_ulonglong,
}
