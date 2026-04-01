//! L4Re-specific raw type definitions.

#![stable(feature = "raw_ext", since = "1.1.0")]
#![deprecated(
    since = "1.8.0",
    note = "these type aliases are no longer supported by \
            the standard library, the `libc` crate on \
            crates.io should be used instead for the correct \
            definitions"
)]
#![allow(deprecated)]

use crate::os::raw::c_ulong;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = u32;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = c_ulong;

#[doc(inline)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub use self::arch::{blkcnt_t, blksize_t, ino_t, nlink_t, off_t, stat, time_t};

#[cfg(any(
    target_arch = "x86",
    target_arch = "m68k",
    target_arch = "csky",
    target_arch = "powerpc",
    target_arch = "sparc",
    target_arch = "arm",
    target_arch = "wasm32"
))]
mod arch {
    use crate::os::raw::{c_long, c_short, c_uint};

    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blkcnt_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blksize_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type ino_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type nlink_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type off_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type time_t = i64;

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad1: c_short,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __st_ino: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mode: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_nlink: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_uid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_gid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_rdev: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad2: c_uint,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blksize: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blocks: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ino: u64,
    }
}

#[cfg(target_arch = "mips")]
mod arch {
    use crate::os::raw::{c_long, c_ulong};

    #[cfg(target_env = "musl")]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blkcnt_t = i64;
    #[cfg(not(target_env = "musl"))]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blkcnt_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blksize_t = u64;
    #[cfg(target_env = "musl")]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type ino_t = u64;
    #[cfg(not(target_env = "musl"))]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type ino_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type nlink_t = u64;
    #[cfg(target_env = "musl")]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type off_t = u64;
    #[cfg(not(target_env = "musl"))]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type off_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type time_t = i64;

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_pad1: [c_long; 3],
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ino: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mode: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_nlink: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_uid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_gid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_rdev: c_ulong,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_pad2: [c_long; 2],
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blksize: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blocks: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_pad5: [c_long; 14],
    }
}

#[cfg(target_arch = "hexagon")]
mod arch {
    use crate::os::raw::{c_int, c_long, c_uint};

    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blkcnt_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blksize_t = c_long;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type ino_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type nlink_t = c_uint;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type off_t = i64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type time_t = i64;

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ino: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mode: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_nlink: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_uid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_gid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_rdev: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad1: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blksize: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad2: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blocks: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad3: [c_int; 2],
    }
}

#[cfg(any(
    target_arch = "mips64",
    target_arch = "s390x",
    target_arch = "sparc64",
    target_arch = "riscv64",
    target_arch = "riscv32"
))]
mod arch {
    pub use libc::{blkcnt_t, blksize_t, ino_t, nlink_t, off_t, stat, time_t};
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use crate::os::raw::{c_int, c_long};

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

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ino: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mode: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_nlink: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_uid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_gid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_rdev: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad1: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blksize: i32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad2: c_int,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blocks: i64,
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

#[cfg(any(target_arch = "x86_64", target_arch = "powerpc64"))]
mod arch {
    use crate::os::raw::{c_int, c_long};

    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blkcnt_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type blksize_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type ino_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type nlink_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type off_t = u64;
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub type time_t = i64;

    #[repr(C)]
    #[derive(Clone)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub struct stat {
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_dev: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ino: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_nlink: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mode: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_uid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_gid: u32,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __pad0: c_int,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_rdev: u64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_size: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blksize: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_blocks: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_atime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_mtime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime: i64,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub st_ctime_nsec: c_long,
        #[stable(feature = "raw_ext", since = "1.1.0")]
        pub __unused: [c_long; 3],
    }
}
