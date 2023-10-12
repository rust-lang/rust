//! AIX specific extensions to primitives in the [`std::fs`] module.
//!
//! [`std::fs`]: crate::fs

#![stable(feature = "metadata_ext", since = "1.1.0")]

use crate::fs::Metadata;
use crate::sys_common::AsInner;

/// OS-specific extensions to [`fs::Metadata`].
///
/// [`fs::Metadata`]: crate::fs::Metadata
#[stable(feature = "metadata_ext", since = "1.1.0")]
pub trait MetadataExt {
    /// Returns the device ID on which this file resides.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::io;
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
    /// use std::os::aix::fs::MetadataExt;
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
        self.as_inner().as_inner().st_atime.tv_sec as i64
    }
    fn st_atime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_atime.tv_nsec as i64
    }
    fn st_mtime(&self) -> i64 {
        self.as_inner().as_inner().st_mtime.tv_sec as i64
    }
    fn st_mtime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_mtime.tv_nsec as i64
    }
    fn st_ctime(&self) -> i64 {
        self.as_inner().as_inner().st_ctime.tv_sec as i64
    }
    fn st_ctime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_ctime.tv_nsec as i64
    }
    fn st_blksize(&self) -> u64 {
        self.as_inner().as_inner().st_blksize as u64
    }
    fn st_blocks(&self) -> u64 {
        self.as_inner().as_inner().st_blocks as u64
    }
}
