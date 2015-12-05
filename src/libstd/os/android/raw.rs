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

#![stable(feature = "raw_ext", since = "1.1.0")]

use os::raw::c_long;

#[unstable(feature = "pthread_t", issue = "29791")] pub type pthread_t = c_long;

#[doc(inline)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub use self::arch::{dev_t, mode_t, blkcnt_t, blksize_t, ino_t, nlink_t, off_t, stat, time_t};

#[cfg(any(target_arch = "arm", target_arch = "x86"))]
mod arch {
    use os::raw::{c_uint, c_uchar, c_ulonglong, c_longlong, c_ulong};
    use os::unix::raw::{uid_t, gid_t};

    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type dev_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type mode_t = u16;

    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blkcnt_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blksize_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type ino_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type nlink_t = u16;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type off_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type time_t = i32;

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: c_ulonglong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad0: [c_uchar; 4],
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __st_ino: ino_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mode: c_uint,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_nlink: c_uint,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_uid: uid_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_gid: gid_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_rdev: c_ulonglong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad3: [c_uchar; 4],
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: c_longlong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blksize: blksize_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blocks: c_ulonglong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime: time_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime_nsec: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime: time_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime_nsec: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime: time_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime_nsec: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ino: c_ulonglong,
    }

}


#[cfg(target_arch = "aarch64")]
mod arch {
    use os::raw::{c_uchar, c_ulong};
    use os::unix::raw::{uid_t, gid_t};

    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type dev_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type mode_t = u32;

    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blkcnt_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blksize_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type ino_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type nlink_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type off_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type time_t = i64;

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: dev_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad0: [c_uchar; 4],
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __st_ino: ino_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mode: mode_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_nlink: nlink_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_uid: uid_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_gid: gid_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_rdev: dev_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad3: [c_uchar; 4],
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: off_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blksize: blksize_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blocks: blkcnt_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime: time_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime_nsec: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime: time_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime_nsec: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime: time_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime_nsec: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ino: ino_t,
    }
}

