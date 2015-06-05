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

#[doc(inline)]
pub use self::arch::{dev_t, mode_t, blkcnt_t, blksize_t, ino_t, nlink_t, off_t, stat, time_t};

#[cfg(target_arch = "arm")]
mod arch {
    use os::raw::{c_uint, c_uchar, c_ulonglong, c_longlong, c_ulong};
    use os::unix::raw::{uid_t, gid_t};

    pub type dev_t = u32;
    pub type mode_t = u16;

    pub type blkcnt_t = u32;
    pub type blksize_t = u32;
    pub type ino_t = u32;
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

}


#[cfg(target_arch = "aarch64")]
mod arch {
    use os::raw::{c_uchar, c_ulong};
    use os::unix::raw::{uid_t, gid_t};

    pub type dev_t = u64;
    pub type mode_t = u32;

    pub type blkcnt_t = u64;
    pub type blksize_t = u32;
    pub type ino_t = u64;
    pub type nlink_t = u32;
    pub type off_t = i64;
    pub type time_t = i64;

    #[repr(C)]
    pub struct stat {
        pub st_dev: dev_t,
        pub __pad0: [c_uchar; 4],
        pub __st_ino: ino_t,
        pub st_mode: mode_t,
        pub st_nlink: nlink_t,
        pub st_uid: uid_t,
        pub st_gid: gid_t,
        pub st_rdev: dev_t,
        pub __pad3: [c_uchar; 4],
        pub st_size: off_t,
        pub st_blksize: blksize_t,
        pub st_blocks: blkcnt_t,
        pub st_atime: time_t,
        pub st_atime_nsec: c_ulong,
        pub st_mtime: time_t,
        pub st_mtime_nsec: c_ulong,
        pub st_ctime: time_t,
        pub st_ctime_nsec: c_ulong,
        pub st_ino: ino_t,
    }

}
