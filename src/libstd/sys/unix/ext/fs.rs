// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unix-specific extensions to primitives in the `std::fs` module.

#![stable(feature = "rust1", since = "1.0.0")]

use fs::{self, Permissions, OpenOptions};
use io;
use libc;
use os::raw::c_long;
use os::unix::raw;
use path::Path;
use sys::fs::MetadataExt as UnixMetadataExt;
use sys;
use sys_common::{FromInner, AsInner, AsInnerMut};

#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const USER_READ: raw::mode_t = 0o400;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const USER_WRITE: raw::mode_t = 0o200;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const USER_EXECUTE: raw::mode_t = 0o100;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const USER_RWX: raw::mode_t = 0o700;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const GROUP_READ: raw::mode_t = 0o040;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const GROUP_WRITE: raw::mode_t = 0o020;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const GROUP_EXECUTE: raw::mode_t = 0o010;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const GROUP_RWX: raw::mode_t = 0o070;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const OTHER_READ: raw::mode_t = 0o004;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const OTHER_WRITE: raw::mode_t = 0o002;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const OTHER_EXECUTE: raw::mode_t = 0o001;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const OTHER_RWX: raw::mode_t = 0o007;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const ALL_READ: raw::mode_t = 0o444;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const ALL_WRITE: raw::mode_t = 0o222;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const ALL_EXECUTE: raw::mode_t = 0o111;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const ALL_RWX: raw::mode_t = 0o777;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const SETUID: raw::mode_t = 0o4000;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const SETGID: raw::mode_t = 0o2000;
#[unstable(feature = "fs_mode", reason = "recently added API", issue = "27712")]
pub const STICKY_BIT: raw::mode_t = 0o1000;

/// Unix-specific extensions to `Permissions`
#[stable(feature = "fs_ext", since = "1.1.0")]
pub trait PermissionsExt {
    /// Returns the underlying raw `mode_t` bits that are the standard Unix
    /// permissions for this file.
    #[stable(feature = "fs_ext", since = "1.1.0")]
    fn mode(&self) -> raw::mode_t;

    /// Sets the underlying raw `mode_t` bits for this set of permissions.
    #[stable(feature = "fs_ext", since = "1.1.0")]
    fn set_mode(&mut self, mode: raw::mode_t);

    /// Creates a new instance of `Permissions` from the given set of Unix
    /// permission bits.
    #[stable(feature = "fs_ext", since = "1.1.0")]
    fn from_mode(mode: raw::mode_t) -> Self;
}

#[stable(feature = "fs_ext", since = "1.1.0")]
impl PermissionsExt for Permissions {
    fn mode(&self) -> raw::mode_t { self.as_inner().mode() }

    fn set_mode(&mut self, mode: raw::mode_t) {
        *self = FromInner::from_inner(FromInner::from_inner(mode));
    }

    fn from_mode(mode: raw::mode_t) -> Permissions {
        FromInner::from_inner(FromInner::from_inner(mode))
    }
}

/// Unix-specific extensions to `OpenOptions`
#[stable(feature = "fs_ext", since = "1.1.0")]
pub trait OpenOptionsExt {
    /// Sets the mode bits that a new file will be created with.
    ///
    /// If a new file is created as part of a `File::open_opts` call then this
    /// specified `mode` will be used as the permission bits for the new file.
    /// If no `mode` is set, the default of `0o666` will be used.
    /// The operating system masks out bits with the systems `umask`, to produce
    /// the final permissions.
    #[stable(feature = "fs_ext", since = "1.1.0")]
    fn mode(&mut self, mode: raw::mode_t) -> &mut Self;
}

#[stable(feature = "fs_ext", since = "1.1.0")]
impl OpenOptionsExt for OpenOptions {
    fn mode(&mut self, mode: raw::mode_t) -> &mut OpenOptions {
        self.as_inner_mut().mode(mode); self
    }
}

// Hm, why are there casts here to the returned type, shouldn't the types always
// be the same? Right you are! Turns out, however, on android at least the types
// in the raw `stat` structure are not the same as the types being returned. Who
// knew!
//
// As a result to make sure this compiles for all platforms we do the manual
// casts and rely on manual lowering to `stat` if the raw type is desired.
#[stable(feature = "metadata_ext", since = "1.1.0")]
pub trait MetadataExt {
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn dev(&self) -> raw::dev_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn ino(&self) -> raw::ino_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn mode(&self) -> raw::mode_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn nlink(&self) -> raw::nlink_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn uid(&self) -> raw::uid_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn gid(&self) -> raw::gid_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn rdev(&self) -> raw::dev_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn size(&self) -> raw::off_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn atime(&self) -> raw::time_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn atime_nsec(&self) -> c_long;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn mtime(&self) -> raw::time_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn mtime_nsec(&self) -> c_long;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn ctime(&self) -> raw::time_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn ctime_nsec(&self) -> c_long;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn blksize(&self) -> raw::blksize_t;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn blocks(&self) -> raw::blkcnt_t;
}

#[stable(feature = "metadata_ext", since = "1.1.0")]
impl MetadataExt for fs::Metadata {
    fn dev(&self) -> raw::dev_t { self.as_raw_stat().st_dev as raw::dev_t }
    fn ino(&self) -> raw::ino_t { self.as_raw_stat().st_ino as raw::ino_t }
    fn mode(&self) -> raw::mode_t { self.as_raw_stat().st_mode as raw::mode_t }
    fn nlink(&self) -> raw::nlink_t { self.as_raw_stat().st_nlink as raw::nlink_t }
    fn uid(&self) -> raw::uid_t { self.as_raw_stat().st_uid as raw::uid_t }
    fn gid(&self) -> raw::gid_t { self.as_raw_stat().st_gid as raw::gid_t }
    fn rdev(&self) -> raw::dev_t { self.as_raw_stat().st_rdev as raw::dev_t }
    fn size(&self) -> raw::off_t { self.as_raw_stat().st_size as raw::off_t }
    fn atime(&self) -> raw::time_t { self.as_raw_stat().st_atime }
    fn atime_nsec(&self) -> c_long { self.as_raw_stat().st_atime_nsec as c_long }
    fn mtime(&self) -> raw::time_t { self.as_raw_stat().st_mtime }
    fn mtime_nsec(&self) -> c_long { self.as_raw_stat().st_mtime_nsec as c_long }
    fn ctime(&self) -> raw::time_t { self.as_raw_stat().st_ctime }
    fn ctime_nsec(&self) -> c_long { self.as_raw_stat().st_ctime_nsec as c_long }

    fn blksize(&self) -> raw::blksize_t {
        self.as_raw_stat().st_blksize as raw::blksize_t
    }
    fn blocks(&self) -> raw::blkcnt_t {
        self.as_raw_stat().st_blocks as raw::blkcnt_t
    }
}

/// Add special unix types (block/char device, fifo and socket)
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
    fn is_block_device(&self) -> bool { self.as_inner().is(libc::S_IFBLK) }
    fn is_char_device(&self) -> bool { self.as_inner().is(libc::S_IFCHR) }
    fn is_fifo(&self) -> bool { self.as_inner().is(libc::S_IFIFO) }
    fn is_socket(&self) -> bool { self.as_inner().is(libc::S_IFSOCK) }
}

/// Unix-specific extension methods for `fs::DirEntry`
#[stable(feature = "dir_entry_ext", since = "1.1.0")]
pub trait DirEntryExt {
    /// Returns the underlying `d_ino` field in the contained `dirent`
    /// structure.
    #[stable(feature = "dir_entry_ext", since = "1.1.0")]
    fn ino(&self) -> raw::ino_t;
}

#[stable(feature = "dir_entry_ext", since = "1.1.0")]
impl DirEntryExt for fs::DirEntry {
    fn ino(&self) -> raw::ino_t { self.as_inner().ino() }
}

/// Creates a new symbolic link on the filesystem.
///
/// The `dst` path will be a symbolic link pointing to the `src` path.
///
/// # Note
///
/// On Windows, you must specify whether a symbolic link points to a file
/// or directory.  Use `os::windows::fs::symlink_file` to create a
/// symbolic link to a file, or `os::windows::fs::symlink_dir` to create a
/// symbolic link to a directory.  Additionally, the process must have
/// `SeCreateSymbolicLinkPrivilege` in order to be able to create a
/// symbolic link.
///
/// # Examples
///
/// ```
/// use std::os::unix::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::symlink("a.txt", "b.txt"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "symlink", since = "1.1.0")]
pub fn symlink<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> io::Result<()>
{
    sys::fs::symlink(src.as_ref(), dst.as_ref())
}

#[stable(feature = "dir_builder", since = "1.6.0")]
/// An extension trait for `fs::DirBuilder` for unix-specific options.
pub trait DirBuilderExt {
    /// Sets the mode to create new directories with. This option defaults to
    /// 0o777.
    #[stable(feature = "dir_builder", since = "1.6.0")]
    fn mode(&mut self, mode: raw::mode_t) -> &mut Self;
}

#[stable(feature = "dir_builder", since = "1.6.0")]
impl DirBuilderExt for fs::DirBuilder {
    fn mode(&mut self, mode: raw::mode_t) -> &mut fs::DirBuilder {
        self.as_inner_mut().set_mode(mode);
        self
    }
}

