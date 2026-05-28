//! Apple-specific raw type definitions
use crate::os::raw::c_long;

#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blkcnt_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type blksize_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type dev_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type ino_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type mode_t = u32;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type nlink_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type off_t = u64;
#[stable(feature = "raw_ext", since = "1.1.0")]
pub type time_t = i64;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = usize;

#[repr(C)]
#[derive(Clone)]
#[stable(feature = "raw_ext", since = "1.1.0")]
#[allow(dead_code)]
pub struct stat {
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_dev: i32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mode: u16,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_nlink: u16,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ino: u64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_uid: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_gid: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_rdev: i32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_atime: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_atime_nsec: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mtime: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_mtime_nsec: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ctime: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_ctime_nsec: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_birthtime: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_birthtime_nsec: c_long,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_size: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blocks: i64,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_blksize: i32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_flags: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_gen: u32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_lspare: i32,
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub st_qspare: [i64; 2],
}
