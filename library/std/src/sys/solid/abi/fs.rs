//! `solid_fs.h`
use crate::os::raw::{c_char, c_int, c_uchar};
pub use libc::{
    blksize_t, dev_t, ino_t, off_t, stat, time_t, O_APPEND, O_CREAT, O_EXCL, O_RDONLY, O_RDWR,
    O_TRUNC, O_WRONLY, SEEK_CUR, SEEK_END, SEEK_SET, S_IEXEC, S_IFBLK, S_IFCHR, S_IFDIR, S_IFIFO,
    S_IFMT, S_IFREG, S_IREAD, S_IWRITE,
};

pub const O_ACCMODE: c_int = 0x3;

pub const SOLID_MAX_PATH: usize = 256;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct dirent {
    pub d_ino: ino_t,
    pub d_type: c_uchar,
    pub d_name: [c_char; 256usize],
}

pub const DT_UNKNOWN: c_uchar = 0;
pub const DT_FIFO: c_uchar = 1;
pub const DT_CHR: c_uchar = 2;
pub const DT_DIR: c_uchar = 4;
pub const DT_BLK: c_uchar = 6;
pub const DT_REG: c_uchar = 8;
pub const DT_LNK: c_uchar = 10;
pub const DT_SOCK: c_uchar = 12;
pub const DT_WHT: c_uchar = 14;

pub type S_DIR = c_int;

extern "C" {
    pub fn SOLID_FS_Open(fd: *mut c_int, path: *const c_char, mode: c_int) -> c_int;
    pub fn SOLID_FS_Close(fd: c_int) -> c_int;
    pub fn SOLID_FS_Read(fd: c_int, buf: *mut u8, size: usize, result: *mut usize) -> c_int;
    pub fn SOLID_FS_Write(fd: c_int, buf: *const u8, size: usize, result: *mut usize) -> c_int;
    pub fn SOLID_FS_Lseek(fd: c_int, offset: off_t, whence: c_int) -> c_int;
    pub fn SOLID_FS_Sync(fd: c_int) -> c_int;
    pub fn SOLID_FS_Ftell(fd: c_int, result: *mut off_t) -> c_int;
    pub fn SOLID_FS_Feof(fd: c_int, result: *mut c_int) -> c_int;
    pub fn SOLID_FS_Fsize(fd: c_int, result: *mut usize) -> c_int;
    pub fn SOLID_FS_Truncate(path: *const c_char, size: off_t) -> c_int;
    pub fn SOLID_FS_OpenDir(path: *const c_char, pDir: *mut S_DIR) -> c_int;
    pub fn SOLID_FS_CloseDir(dir: S_DIR) -> c_int;
    pub fn SOLID_FS_ReadDir(dir: S_DIR, dirp: *mut dirent) -> c_int;
    pub fn SOLID_FS_Stat(path: *const c_char, buf: *mut stat) -> c_int;
    pub fn SOLID_FS_Unlink(path: *const c_char) -> c_int;
    pub fn SOLID_FS_Rename(oldpath: *const c_char, newpath: *const c_char) -> c_int;
    pub fn SOLID_FS_Chmod(path: *const c_char, mode: c_int) -> c_int;
    pub fn SOLID_FS_Utime(path: *const c_char, time: time_t) -> c_int;
    pub fn SOLID_FS_Mkdir(path: *const c_char) -> c_int;
}
