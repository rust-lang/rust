// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Linux-specific raw type definitions

#![stable(feature = "raw_ext", since = "1.1.0")]

use os::raw::c_ulong;

#[stable(feature = "raw_ext", since = "1.1.0")] pub type dev_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")] pub type mode_t = u32;

#[unstable(feature = "pthread_t", issue = "29791")] pub type pthread_t = c_ulong;

#[doc(inline)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub use self::arch::{off_t, ino_t, nlink_t, blksize_t, blkcnt_t, stat, time_t};

#[cfg(any(target_arch = "x86",
          target_arch = "le32",
          target_arch = "powerpc",
          target_arch = "arm",
          target_arch = "asmjs"))]
mod arch {
    use super::{dev_t, mode_t};
    use os::raw::{c_long, c_short};
    use os::unix::raw::{gid_t, uid_t};

    #[stable(feature = "raw_ext", since = "1.1.0")] pub type blkcnt_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type blksize_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type ino_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type nlink_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type off_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type time_t = i32;

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: dev_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad1: c_short,
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
        pub st_rdev: dev_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad2: c_short,
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
        pub __unused4: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __unused5: c_long,
    }
}

#[cfg(any(target_arch = "mips",
          target_arch = "mipsel"))]
mod arch {
    use super::mode_t;
    use os::raw::{c_long, c_ulong};
    use os::unix::raw::{gid_t, uid_t};

    #[stable(feature = "raw_ext", since = "1.1.0")] pub type blkcnt_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type blksize_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type ino_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type nlink_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type off_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type time_t = i32;

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_pad1: [c_long; 3],
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
        pub st_rdev: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_pad2: [c_long; 2],
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: off_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_pad3: c_long,
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
        pub st_blksize: blksize_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blocks: blkcnt_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_pad5: [c_long; 14],
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use super::{dev_t, mode_t};
    use os::raw::{c_long, c_int};
    use os::unix::raw::{gid_t, uid_t};

    #[stable(feature = "raw_ext", since = "1.1.0")] pub type blkcnt_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type blksize_t = i32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type ino_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type nlink_t = u32;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type off_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type time_t = i64;

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
        pub st_rdev: dev_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad1: dev_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: off_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blksize: blksize_t,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad2: c_int,
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
        pub __unused: [c_int; 2],
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use super::{dev_t, mode_t};
    use os::raw::{c_long, c_int};
    use os::unix::raw::{gid_t, uid_t};

    #[stable(feature = "raw_ext", since = "1.1.0")] pub type blkcnt_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type blksize_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type ino_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type nlink_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type off_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")] pub type time_t = i64;

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
        pub __pad0: c_int,
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
        pub __unused: [c_long; 3],
    }
}
