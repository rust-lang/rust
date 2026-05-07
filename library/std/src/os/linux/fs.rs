//! Linux-specific extensions to primitives in the [`std::fs`] module.
//!
//! [`std::fs`]: crate::fs

#![stable(feature = "metadata_ext", since = "1.1.0")]

use core::mem;

use crate::fs::Metadata;
#[allow(deprecated)]
use crate::os::linux::raw;
use crate::os::raw::c_void;
use crate::sys::AsInner;
use crate::sys::fs::cfg_has_statx;
cfg_has_statx! {{
    use crate::sys::fs::{FileAttr, StatxExtraFields};
    use crate::sys::FromInner;
} else {
    use crate::sys::unsupported;
}}

/// OS-specific extensions to [`fs::Metadata`].
///
/// [`fs::Metadata`]: crate::fs::Metadata
#[stable(feature = "metadata_ext", since = "1.1.0")]
pub trait MetadataExt {
    /// Gain a reference to the underlying `stat` structure which contains
    /// the raw information returned by the OS.
    ///
    /// The contents of the returned [`stat`] are **not** consistent across
    /// Unix platforms. The `os::unix::fs::MetadataExt` trait contains the
    /// cross-Unix abstractions contained within the raw stat.
    ///
    /// [`stat`]: struct@crate::os::linux::raw::stat
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     let stat = meta.as_raw_stat();
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    #[deprecated(since = "1.8.0", note = "other methods of this trait are now preferred")]
    #[allow(deprecated)]
    fn as_raw_stat(&self) -> &raw::stat;

    /// Creates a [`Metadata`] from a const void pointer populated by the [`statx`] syscall.
    ///
    /// # Safety
    ///
    /// The caller must take care to provide a valid const void pointer containing information
    /// populated by the [`statx`] syscall.
    ///
    /// [`Metadata`]: crate::fs::Metadata
    /// [`statx`]: https://docs.rs/libc/latest/libc/struct.statx.html
    ///
    /// ```no_run
    /// #![feature(metadata_statx)]
    /// use libc::statx;
    /// use std::ffi::c_void;
    /// use std::fs::{write, Metadata};
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     write("hello.txt", "Hello World!")?;
    ///     let mut buf = Box::<statx>::new_uninit();
    ///     unsafe {
    ///         libc::statx(
    ///             libc::AT_FDCWD,
    ///             "hello.txt".as_ptr().cast(),
    ///             libc::AT_STATX_SYNC_AS_STAT,
    ///             libc::STATX_BASIC_STATS,
    ///             buf.as_mut_ptr().cast()
    ///         );
    ///     }
    ///     let statxbuf: Box<statx> = unsafe { buf.assume_init() };
    ///     let metadata = unsafe { Metadata::from_statx(&*statxbuf as *const statx as *const c_void) };
    ///     assert_eq!(metadata.len(), 12); // "Hello World!" is 12 bytes
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "metadata_statx", issue = "156268")]
    unsafe fn from_statx(statxbuf: *const c_void) -> Self;

    /// Returns the device ID on which this file resides.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_dev());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_dev(&self) -> u64;
    /// Returns the inode number.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_ino());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_ino(&self) -> u64;
    /// Returns the file type and mode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_mode());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_mode(&self) -> u32;
    /// Returns the number of hard links to file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_nlink());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_nlink(&self) -> u64;
    /// Returns the user ID of the file owner.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_uid());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_uid(&self) -> u32;
    /// Returns the group ID of the file owner.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_gid());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_gid(&self) -> u32;
    /// Returns the device ID that this file represents. Only relevant for special file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_rdev());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_rdev(&self) -> u64;
    /// Returns the size of the file (if it is a regular file or a symbolic link) in bytes.
    ///
    /// The size of a symbolic link is the length of the pathname it contains,
    /// without a terminating null byte.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_size());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_size(&self) -> u64;
    /// Returns the last access time of the file, in seconds since Unix Epoch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_atime());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_atime(&self) -> i64;
    /// Returns the last access time of the file, in nanoseconds since [`st_atime`].
    ///
    /// [`st_atime`]: Self::st_atime
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_atime_nsec());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_atime_nsec(&self) -> i64;
    /// Returns the last modification time of the file, in seconds since Unix Epoch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_mtime());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_mtime(&self) -> i64;
    /// Returns the last modification time of the file, in nanoseconds since [`st_mtime`].
    ///
    /// [`st_mtime`]: Self::st_mtime
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_mtime_nsec());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_mtime_nsec(&self) -> i64;
    /// Returns the last status change time of the file, in seconds since Unix Epoch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_ctime());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_ctime(&self) -> i64;
    /// Returns the last status change time of the file, in nanoseconds since [`st_ctime`].
    ///
    /// [`st_ctime`]: Self::st_ctime
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_ctime_nsec());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_ctime_nsec(&self) -> i64;
    /// Returns the "preferred" block size for efficient filesystem I/O.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_blksize());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_blksize(&self) -> u64;
    /// Returns the number of blocks allocated to the file, 512-byte units.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::linux::fs::MetadataExt;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata("some_file")?;
    ///     println!("{}", meta.st_blocks());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "metadata_ext2", since = "1.8.0")]
    fn st_blocks(&self) -> u64;
}

#[stable(feature = "metadata_ext", since = "1.1.0")]
impl MetadataExt for Metadata {
    #[allow(deprecated)]
    fn as_raw_stat(&self) -> &raw::stat {
        #[cfg(target_env = "musl")]
        unsafe {
            &*(self.as_inner().as_inner() as *const libc::stat as *const raw::stat)
        }
        #[cfg(not(target_env = "musl"))]
        unsafe {
            &*(self.as_inner().as_inner() as *const libc::stat64 as *const raw::stat)
        }
    }
    cfg_has_statx! {{
        unsafe fn from_statx(statxbuf: *const c_void) -> Metadata {
            let buf =  statxbuf as *const libc::statx;

            // We cannot fill `stat64` exhaustively because of private padding fields.
            let mut stat: libc::stat64 = mem::zeroed();
            // `c_ulong` on gnu-mips, `dev_t` otherwise
            stat.st_dev = libc::makedev((*buf).stx_dev_major, (*buf).stx_dev_minor) as _;
            stat.st_ino = (*buf).stx_ino as libc::ino64_t;
            stat.st_nlink = (*buf).stx_nlink as libc::nlink_t;
            stat.st_mode = (*buf).stx_mode as libc::mode_t;
            stat.st_uid = (*buf).stx_uid as libc::uid_t;
            stat.st_gid = (*buf).stx_gid as libc::gid_t;
            stat.st_rdev = libc::makedev((*buf).stx_rdev_major, (*buf).stx_rdev_minor) as _;
            stat.st_size = (*buf).stx_size as libc::off64_t;
            stat.st_blksize = (*buf).stx_blksize as libc::blksize_t;
            stat.st_blocks = (*buf).stx_blocks as libc::blkcnt64_t;
            stat.st_atime = (*buf).stx_atime.tv_sec as libc::time_t;
            // `i64` on gnu-x86_64-x32, `c_ulong` otherwise.
            stat.st_atime_nsec = (*buf).stx_atime.tv_nsec as _;
            stat.st_mtime = (*buf).stx_mtime.tv_sec as libc::time_t;
            stat.st_mtime_nsec = (*buf).stx_mtime.tv_nsec as _;
            stat.st_ctime = (*buf).stx_ctime.tv_sec as libc::time_t;
            stat.st_ctime_nsec = (*buf).stx_ctime.tv_nsec as _;

            let extra = StatxExtraFields::from_statx(statxbuf);

            Metadata::from_inner(FileAttr::from_statx(stat, Some(extra)))
        }} else {
            unsafe fn from_statx(statxbuf: *const c_void) -> Self {
                unsupported();
            }
    }}
    fn st_dev(&self) -> u64 {
        self.as_inner().as_inner().st_dev as u64
    }
    fn st_ino(&self) -> u64 {
        self.as_inner().as_inner().st_ino as u64
    }
    fn st_mode(&self) -> u32 {
        self.as_inner().as_inner().st_mode as u32
    }
    fn st_nlink(&self) -> u64 {
        self.as_inner().as_inner().st_nlink as u64
    }
    fn st_uid(&self) -> u32 {
        self.as_inner().as_inner().st_uid as u32
    }
    fn st_gid(&self) -> u32 {
        self.as_inner().as_inner().st_gid as u32
    }
    fn st_rdev(&self) -> u64 {
        self.as_inner().as_inner().st_rdev as u64
    }
    fn st_size(&self) -> u64 {
        self.as_inner().as_inner().st_size as u64
    }
    fn st_atime(&self) -> i64 {
        let file_attr = self.as_inner();
        #[cfg(all(target_env = "gnu", target_pointer_width = "32"))]
        if let Some(atime) = file_attr.stx_atime() {
            return atime.tv_sec;
        }
        file_attr.as_inner().st_atime as i64
    }
    fn st_atime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_atime_nsec as i64
    }
    fn st_mtime(&self) -> i64 {
        let file_attr = self.as_inner();
        #[cfg(all(target_env = "gnu", target_pointer_width = "32"))]
        if let Some(mtime) = file_attr.stx_mtime() {
            return mtime.tv_sec;
        }
        file_attr.as_inner().st_mtime as i64
    }
    fn st_mtime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_mtime_nsec as i64
    }
    fn st_ctime(&self) -> i64 {
        let file_attr = self.as_inner();
        #[cfg(all(target_env = "gnu", target_pointer_width = "32"))]
        if let Some(ctime) = file_attr.stx_ctime() {
            return ctime.tv_sec;
        }
        file_attr.as_inner().st_ctime as i64
    }
    fn st_ctime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_ctime_nsec as i64
    }
    fn st_blksize(&self) -> u64 {
        self.as_inner().as_inner().st_blksize as u64
    }
    fn st_blocks(&self) -> u64 {
        self.as_inner().as_inner().st_blocks as u64
    }
}
