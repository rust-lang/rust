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

use prelude::v1::*;

use fs::{self, Permissions, OpenOptions};
use io;
use mem;
use os::raw::c_long;
use os::unix::raw;
use path::Path;
use sys::platform;
use sys;
use sys_common::{FromInner, AsInner, AsInnerMut};

#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const USER_READ: raw::mode_t = 0o400;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const USER_WRITE: raw::mode_t = 0o200;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const USER_EXECUTE: raw::mode_t = 0o100;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const USER_RWX: raw::mode_t = 0o700;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const GROUP_READ: raw::mode_t = 0o040;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const GROUP_WRITE: raw::mode_t = 0o020;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const GROUP_EXECUTE: raw::mode_t = 0o010;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const GROUP_RWX: raw::mode_t = 0o070;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const OTHER_READ: raw::mode_t = 0o004;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const OTHER_WRITE: raw::mode_t = 0o002;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const OTHER_EXECUTE: raw::mode_t = 0o001;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const OTHER_RWX: raw::mode_t = 0o007;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const ALL_READ: raw::mode_t = 0o444;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const ALL_WRITE: raw::mode_t = 0o222;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const ALL_EXECUTE: raw::mode_t = 0o111;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const ALL_RWX: raw::mode_t = 0o777;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const SETUID: raw::mode_t = 0o4000;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const SETGID: raw::mode_t = 0o2000;
#[unstable(feature = "fs_mode", reason = "recently added API")]
pub const STICKY_BIT: raw::mode_t = 0o1000;

/// Unix-specific extensions to `Permissions`
#[unstable(feature = "fs_ext",
           reason = "may want a more useful mode abstraction")]
pub trait PermissionsExt {
    fn mode(&self) -> raw::mode_t;
    fn set_mode(&mut self, mode: raw::mode_t);
    fn from_mode(mode: raw::mode_t) -> Self;
}

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
#[unstable(feature = "fs_ext",
           reason = "may want a more useful mode abstraction")]
pub trait OpenOptionsExt {
    /// Sets the mode bits that a new file will be created with.
    ///
    /// If a new file is created as part of a `File::open_opts` call then this
    /// specified `mode` will be used as the permission bits for the new file.
    fn mode(&mut self, mode: raw::mode_t) -> &mut Self;
}

impl OpenOptionsExt for OpenOptions {
    fn mode(&mut self, mode: raw::mode_t) -> &mut OpenOptions {
        self.as_inner_mut().mode(mode); self
    }
}

#[unstable(feature = "metadata_ext", reason = "recently added API")]
pub struct Metadata(sys::fs::FileAttr);

#[unstable(feature = "metadata_ext", reason = "recently added API")]
pub trait MetadataExt {
    fn as_raw(&self) -> &Metadata;
}

impl MetadataExt for fs::Metadata {
    fn as_raw(&self) -> &Metadata {
        let inner: &sys::fs::FileAttr = self.as_inner();
        unsafe { mem::transmute(inner) }
    }
}

impl AsInner<platform::raw::stat> for Metadata {
    fn as_inner(&self) -> &platform::raw::stat { self.0.as_inner() }
}

// Hm, why are there casts here to the returned type, shouldn't the types always
// be the same? Right you are! Turns out, however, on android at least the types
// in the raw `stat` structure are not the same as the types being returned. Who
// knew!
//
// As a result to make sure this compiles for all platforms we do the manual
// casts and rely on manual lowering to `stat` if the raw type is desired.
#[unstable(feature = "metadata_ext", reason = "recently added API")]
impl Metadata {
    pub fn dev(&self) -> raw::dev_t { self.0.raw().st_dev as raw::dev_t }
    pub fn ino(&self) -> raw::ino_t { self.0.raw().st_ino as raw::ino_t }
    pub fn mode(&self) -> raw::mode_t { self.0.raw().st_mode as raw::mode_t }
    pub fn nlink(&self) -> raw::nlink_t { self.0.raw().st_nlink as raw::nlink_t }
    pub fn uid(&self) -> raw::uid_t { self.0.raw().st_uid as raw::uid_t }
    pub fn gid(&self) -> raw::gid_t { self.0.raw().st_gid as raw::gid_t }
    pub fn rdev(&self) -> raw::dev_t { self.0.raw().st_rdev as raw::dev_t }
    pub fn size(&self) -> raw::off_t { self.0.raw().st_size as raw::off_t }
    pub fn atime(&self) -> raw::time_t { self.0.raw().st_atime }
    pub fn atime_nsec(&self) -> c_long { self.0.raw().st_atime_nsec as c_long }
    pub fn mtime(&self) -> raw::time_t { self.0.raw().st_mtime }
    pub fn mtime_nsec(&self) -> c_long { self.0.raw().st_mtime_nsec as c_long }
    pub fn ctime(&self) -> raw::time_t { self.0.raw().st_ctime }
    pub fn ctime_nsec(&self) -> c_long { self.0.raw().st_ctime_nsec as c_long }

    pub fn blksize(&self) -> raw::blksize_t {
        self.0.raw().st_blksize as raw::blksize_t
    }
    pub fn blocks(&self) -> raw::blkcnt_t {
        self.0.raw().st_blocks as raw::blkcnt_t
    }
}

#[unstable(feature = "dir_entry_ext", reason = "recently added API")]
pub trait DirEntryExt {
    fn ino(&self) -> raw::ino_t;
}

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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn symlink<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> io::Result<()>
{
    sys::fs::symlink(src.as_ref(), dst.as_ref())
}

#[unstable(feature = "dir_builder", reason = "recently added API")]
/// An extension trait for `fs::DirBuilder` for unix-specific options.
pub trait DirBuilderExt {
    /// Sets the mode to create new directories with. This option defaults to
    /// 0o777.
    fn mode(&mut self, mode: raw::mode_t) -> &mut Self;
}

impl DirBuilderExt for fs::DirBuilder {
    fn mode(&mut self, mode: raw::mode_t) -> &mut fs::DirBuilder {
        self.as_inner_mut().set_mode(mode);
        self
    }
}

