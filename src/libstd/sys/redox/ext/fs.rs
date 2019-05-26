//! Redox-specific extensions to primitives in the `std::fs` module.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::fs::{self, Permissions, OpenOptions};
use crate::io;
use crate::path::Path;
use crate::sys;
use crate::sys_common::{FromInner, AsInner, AsInnerMut};

/// Redox-specific extensions to [`fs::Permissions`].
///
/// [`fs::Permissions`]: ../../../../std/fs/struct.Permissions.html
#[stable(feature = "fs_ext", since = "1.1.0")]
pub trait PermissionsExt {
    /// Returns the underlying raw `mode_t` bits that are the standard Redox
    /// permissions for this file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::os::redox::fs::PermissionsExt;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::create("foo.txt")?;
    ///     let metadata = f.metadata()?;
    ///     let permissions = metadata.permissions();
    ///
    ///     println!("permissions: {:o}", permissions.mode());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "fs_ext", since = "1.1.0")]
    fn mode(&self) -> u32;

    /// Sets the underlying raw bits for this set of permissions.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::os::redox::fs::PermissionsExt;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::create("foo.txt")?;
    ///     let metadata = f.metadata()?;
    ///     let mut permissions = metadata.permissions();
    ///
    ///     permissions.set_mode(0o644); // Read/write for owner and read for others.
    ///     assert_eq!(permissions.mode(), 0o644);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "fs_ext", since = "1.1.0")]
    fn set_mode(&mut self, mode: u32);

    /// Creates a new instance of `Permissions` from the given set of Redox
    /// permission bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::Permissions;
    /// use std::os::redox::fs::PermissionsExt;
    ///
    /// // Read/write for owner and read for others.
    /// let permissions = Permissions::from_mode(0o644);
    /// assert_eq!(permissions.mode(), 0o644);
    /// ```
    #[stable(feature = "fs_ext", since = "1.1.0")]
    fn from_mode(mode: u32) -> Self;
}

#[stable(feature = "fs_ext", since = "1.1.0")]
impl PermissionsExt for Permissions {
    fn mode(&self) -> u32 {
        self.as_inner().mode()
    }

    fn set_mode(&mut self, mode: u32) {
        *self = Permissions::from_inner(FromInner::from_inner(mode));
    }

    fn from_mode(mode: u32) -> Permissions {
        Permissions::from_inner(FromInner::from_inner(mode))
    }
}

/// Redox-specific extensions to [`fs::OpenOptions`].
///
/// [`fs::OpenOptions`]: ../../../../std/fs/struct.OpenOptions.html
#[stable(feature = "fs_ext", since = "1.1.0")]
pub trait OpenOptionsExt {
    /// Sets the mode bits that a new file will be created with.
    ///
    /// If a new file is created as part of a `File::open_opts` call then this
    /// specified `mode` will be used as the permission bits for the new file.
    /// If no `mode` is set, the default of `0o666` will be used.
    /// The operating system masks out bits with the systems `umask`, to produce
    /// the final permissions.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #![feature(libc)]
    /// extern crate libc;
    /// use std::fs::OpenOptions;
    /// use std::os::redox::fs::OpenOptionsExt;
    ///
    /// # fn main() {
    /// let mut options = OpenOptions::new();
    /// options.mode(0o644); // Give read/write for owner and read for others.
    /// let file = options.open("foo.txt");
    /// # }
    /// ```
    #[stable(feature = "fs_ext", since = "1.1.0")]
    fn mode(&mut self, mode: u32) -> &mut Self;

    /// Passes custom flags to the `flags` argument of `open`.
    ///
    /// The bits that define the access mode are masked out with `O_ACCMODE`, to
    /// ensure they do not interfere with the access mode set by Rusts options.
    ///
    /// Custom flags can only set flags, not remove flags set by Rusts options.
    /// This options overwrites any previously set custom flags.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #![feature(libc)]
    /// extern crate libc;
    /// use std::fs::OpenOptions;
    /// use std::os::redox::fs::OpenOptionsExt;
    ///
    /// # fn main() {
    /// let mut options = OpenOptions::new();
    /// options.write(true);
    /// if cfg!(target_os = "redox") {
    ///     options.custom_flags(libc::O_NOFOLLOW);
    /// }
    /// let file = options.open("foo.txt");
    /// # }
    /// ```
    #[stable(feature = "open_options_ext", since = "1.10.0")]
    fn custom_flags(&mut self, flags: i32) -> &mut Self;
}

#[stable(feature = "fs_ext", since = "1.1.0")]
impl OpenOptionsExt for OpenOptions {
    fn mode(&mut self, mode: u32) -> &mut OpenOptions {
        self.as_inner_mut().mode(mode); self
    }

    fn custom_flags(&mut self, flags: i32) -> &mut OpenOptions {
        self.as_inner_mut().custom_flags(flags); self
    }
}

/// Redox-specific extensions to [`fs::Metadata`].
///
/// [`fs::Metadata`]: ../../../../std/fs/struct.Metadata.html
#[stable(feature = "metadata_ext", since = "1.1.0")]
pub trait MetadataExt {
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn dev(&self) -> u64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn ino(&self) -> u64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn mode(&self) -> u32;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn nlink(&self) -> u64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn uid(&self) -> u32;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn gid(&self) -> u32;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn size(&self) -> u64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn atime(&self) -> i64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn atime_nsec(&self) -> i64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn mtime(&self) -> i64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn mtime_nsec(&self) -> i64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn ctime(&self) -> i64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn ctime_nsec(&self) -> i64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn blksize(&self) -> u64;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn blocks(&self) -> u64;
}

// Hm, why are there casts here to the returned type, shouldn't the types always
// be the same? Right you are! Turns out, however, on android at least the types
// in the raw `stat` structure are not the same as the types being returned. Who
// knew!
//
// As a result to make sure this compiles for all platforms we do the manual
// casts and rely on manual lowering to `stat` if the raw type is desired.
#[stable(feature = "metadata_ext", since = "1.1.0")]
impl MetadataExt for fs::Metadata {
    fn dev(&self) -> u64 {
        self.as_inner().as_inner().st_dev as u64
    }
    fn ino(&self) -> u64 {
        self.as_inner().as_inner().st_ino as u64
    }
    fn mode(&self) -> u32 {
        self.as_inner().as_inner().st_mode as u32
    }
    fn nlink(&self) -> u64 {
        self.as_inner().as_inner().st_nlink as u64
    }
    fn uid(&self) -> u32 {
        self.as_inner().as_inner().st_uid as u32
    }
    fn gid(&self) -> u32 {
        self.as_inner().as_inner().st_gid as u32
    }
    fn size(&self) -> u64 {
        self.as_inner().as_inner().st_size as u64
    }
    fn atime(&self) -> i64 {
        self.as_inner().as_inner().st_atime as i64
    }
    fn atime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_atime_nsec as i64
    }
    fn mtime(&self) -> i64 {
        self.as_inner().as_inner().st_mtime as i64
    }
    fn mtime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_mtime_nsec as i64
    }
    fn ctime(&self) -> i64 {
        self.as_inner().as_inner().st_ctime as i64
    }
    fn ctime_nsec(&self) -> i64 {
        self.as_inner().as_inner().st_ctime_nsec as i64
    }
    fn blksize(&self) -> u64 {
        self.as_inner().as_inner().st_blksize as u64
    }
    fn blocks(&self) -> u64 {
        self.as_inner().as_inner().st_blocks as u64
    }
}

/// Redox-specific extensions for [`FileType`].
///
/// Adds support for special Unix file types such as block/character devices,
/// pipes, and sockets.
///
/// [`FileType`]: ../../../../std/fs/struct.FileType.html
#[stable(feature = "file_type_ext", since = "1.5.0")]
pub trait FileTypeExt {
    /// Returns whether this file type is a block device.
    #[stable(feature = "file_type_ext", since = "1.5.0")]
    fn is_block_device(&self) -> bool;
    /// Returns whether this file type is a char device.
    #[stable(feature = "file_type_ext", since = "1.5.0")]
    fn is_char_device(&self) -> bool;
    /// Returns whether this file type is a fifo.
    #[stable(feature = "file_type_ext", since = "1.5.0")]
    fn is_fifo(&self) -> bool;
    /// Returns whether this file type is a socket.
    #[stable(feature = "file_type_ext", since = "1.5.0")]
    fn is_socket(&self) -> bool;
}

#[stable(feature = "file_type_ext", since = "1.5.0")]
impl FileTypeExt for fs::FileType {
    fn is_block_device(&self) -> bool { false /*FIXME: Implement block device mode*/ }
    fn is_char_device(&self) -> bool { false /*FIXME: Implement char device mode*/ }
    fn is_fifo(&self) -> bool { false /*FIXME: Implement fifo mode*/ }
    fn is_socket(&self) -> bool { false /*FIXME: Implement socket mode*/ }
}

/// Creates a new symbolic link on the filesystem.
///
/// The `dst` path will be a symbolic link pointing to the `src` path.
///
/// # Note
///
/// On Windows, you must specify whether a symbolic link points to a file
/// or directory. Use `os::windows::fs::symlink_file` to create a
/// symbolic link to a file, or `os::windows::fs::symlink_dir` to create a
/// symbolic link to a directory. Additionally, the process must have
/// `SeCreateSymbolicLinkPrivilege` in order to be able to create a
/// symbolic link.
///
/// # Examples
///
/// ```no_run
/// use std::os::redox::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::symlink("a.txt", "b.txt")?;
///     Ok(())
/// }
/// ```
#[stable(feature = "symlink", since = "1.1.0")]
pub fn symlink<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> io::Result<()>
{
    sys::fs::symlink(src.as_ref(), dst.as_ref())
}

/// Redox-specific extensions to [`fs::DirBuilder`].
///
/// [`fs::DirBuilder`]: ../../../../std/fs/struct.DirBuilder.html
#[stable(feature = "dir_builder", since = "1.6.0")]
pub trait DirBuilderExt {
    /// Sets the mode to create new directories with. This option defaults to
    /// 0o777.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::DirBuilder;
    /// use std::os::redox::fs::DirBuilderExt;
    ///
    /// let mut builder = DirBuilder::new();
    /// builder.mode(0o755);
    /// ```
    #[stable(feature = "dir_builder", since = "1.6.0")]
    fn mode(&mut self, mode: u32) -> &mut Self;
}

#[stable(feature = "dir_builder", since = "1.6.0")]
impl DirBuilderExt for fs::DirBuilder {
    fn mode(&mut self, mode: u32) -> &mut fs::DirBuilder {
        self.as_inner_mut().set_mode(mode);
        self
    }
}
