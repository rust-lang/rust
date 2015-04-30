// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Nacl-specific raw type definitions

pub type dev_t = u64;
pub type mode_t = u32;

pub use self::arch::{off_t, ino_t, nlink_t, blksize_t, blkcnt_t, stat, time_t};

#[cfg(any(target_arch = "x86",
          target_arch = "le32",
          target_arch = "powerpc",
          target_arch = "arm"))]
mod arch {
    use super::{dev_t, mode_t};
    use os::raw::{c_long, c_short};
    use os::unix::raw::{gid_t, uid_t};

    pub type blkcnt_t = i32;
    pub type blksize_t = i32;
    pub type ino_t = u32;
    pub type nlink_t = u32;
    pub type off_t = i32;
    pub type time_t = i32;

    #[repr(C)]
    pub struct stat {
        pub st_dev: dev_t,
        pub __pad1: c_short,
        pub st_ino: ino_t,
        pub st_mode: mode_t,
        pub st_nlink: nlink_t,
        pub st_uid: uid_t,
        pub st_gid: gid_t,
        pub st_rdev: dev_t,
        pub __pad2: c_short,
        pub st_size: off_t,
        pub st_blksize: blksize_t,
        pub st_blocks: blkcnt_t,
        pub st_atime: time_t,
        pub st_atime_nsec: c_long,
        pub st_mtime: time_t,
        pub st_mtime_nsec: c_long,
        pub st_ctime: time_t,
        pub st_ctime_nsec: c_long,
        pub __unused4: c_long,
        pub __unused5: c_long,
    }
}

#[cfg(any(target_arch = "mips",
          target_arch = "mipsel"))]
mod arch {
    use super::{dev_t, mode_t};
    use os::raw::c_long;
    use os::unix::raw::{gid_t, uid_t};

    pub type blkcnt_t = i32;
    pub type blksize_t = i32;
    pub type ino_t = u32;
    pub type nlink_t = u32;
    pub type off_t = i32;
    pub type time_t = i32;

    #[repr(C)]
    pub struct stat {
        pub st_dev: c_ulong,
        pub st_pad1: [c_long; 3],
        pub st_ino: ino_t,
        pub st_mode: mode_t,
        pub st_nlink: nlink_t,
        pub st_uid: uid_t,
        pub st_gid: gid_t,
        pub st_rdev: c_ulong,
        pub st_pad2: [c_long; 2],
        pub st_size: off_t,
        pub st_pad3: c_long,
        pub st_atime: time_t,
        pub st_atime_nsec: c_long,
        pub st_mtime: time_t,
        pub st_mtime_nsec: c_long,
        pub st_ctime: time_t,
        pub st_ctime_nsec: c_long,
        pub st_blksize: blksize_t,
        pub st_blocks: blkcnt_t,
        pub st_pad5: [c_long; 14],
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use super::{dev_t, mode_t};
    use os::raw::{c_long, c_int};
    use os::unix::raw::{gid_t, uid_t};

    pub type blkcnt_t = i64;
    pub type blksize_t = i32;
    pub type ino_t = u64;
    pub type nlink_t = u32;
    pub type off_t = i64;
    pub type time_t = i64;

    #[repr(C)]
    pub struct stat {
        pub st_dev: dev_t,
        pub st_ino: ino_t,
        pub st_mode: mode_t,
        pub st_nlink: nlink_t,
        pub st_uid: uid_t,
        pub st_gid: gid_t,
        pub st_rdev: dev_t,
        pub __pad1: dev_t,
        pub st_size: off_t,
        pub st_blksize: blksize_t,
        pub __pad2: c_int,
        pub st_blocks: blkcnt_t,
        pub st_atime: time_t,
        pub st_atime_nsec: c_long,
        pub st_mtime: time_t,
        pub st_mtime_nsec: c_long,
        pub st_ctime: time_t,
        pub st_ctime_nsec: c_long,
        pub __unused: [c_int; 2],
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use super::{dev_t, mode_t};
    use os::raw::{c_long, c_int};
    use os::unix::raw::{gid_t, uid_t};

    pub type blkcnt_t = i64;
    pub type blksize_t = i64;
    pub type ino_t = u64;
    pub type nlink_t = u64;
    pub type off_t = i64;
    pub type time_t = i64;

    #[repr(C)]
    pub struct stat {
        pub st_dev: dev_t,
        pub st_ino: ino_t,
        pub st_nlink: nlink_t,
        pub st_mode: mode_t,
        pub st_uid: uid_t,
        pub st_gid: gid_t,
        pub __pad0: c_int,
        pub st_rdev: dev_t,
        pub st_size: off_t,
        pub st_blksize: blksize_t,
        pub st_blocks: blkcnt_t,
        pub st_atime: time_t,
        pub st_atime_nsec: c_long,
        pub st_mtime: time_t,
        pub st_mtime_nsec: c_long,
        pub st_ctime: time_t,
        pub st_ctime_nsec: c_long,
        pub __unused: [c_long; 3],
    }
}
