// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows-specific extensions for the primitives in `std::fs`

#![stable(feature = "rust1", since = "1.0.0")]

use fs::{OpenOptions, Metadata};
use io;
use path::Path;
use sys;
use sys::inner::{AsInnerMut, AsInner};

/// Windows-specific extensions to `OpenOptions`
#[unstable(feature = "open_options_ext",
           reason = "may require more thought/methods",
           issue = "27720")]
pub trait OpenOptionsExt {
    /// Overrides the `dwDesiredAccess` argument to the call to `CreateFile`
    /// with the specified value.
    fn desired_access(&mut self, access: u32) -> &mut Self;

    /// Overrides the `dwCreationDisposition` argument to the call to
    /// `CreateFile` with the specified value.
    ///
    /// This will override any values of the standard `create` flags, for
    /// example.
    fn creation_disposition(&mut self, val: u32) -> &mut Self;

    /// Overrides the `dwFlagsAndAttributes` argument to the call to
    /// `CreateFile` with the specified value.
    ///
    /// This will override any values of the standard flags on the
    /// `OpenOptions` structure.
    fn flags_and_attributes(&mut self, val: u32) -> &mut Self;

    /// Overrides the `dwShareMode` argument to the call to `CreateFile` with
    /// the specified value.
    ///
    /// This will override any values of the standard flags on the
    /// `OpenOptions` structure.
    fn share_mode(&mut self, val: u32) -> &mut Self;
}

impl OpenOptionsExt for OpenOptions {
    fn desired_access(&mut self, access: u32) -> &mut OpenOptions {
        self.as_inner_mut().desired_access(access); self
    }
    fn creation_disposition(&mut self, access: u32) -> &mut OpenOptions {
        self.as_inner_mut().creation_disposition(access); self
    }
    fn flags_and_attributes(&mut self, access: u32) -> &mut OpenOptions {
        self.as_inner_mut().flags_and_attributes(access); self
    }
    fn share_mode(&mut self, access: u32) -> &mut OpenOptions {
        self.as_inner_mut().share_mode(access); self
    }
}

/// Extension methods for `fs::Metadata` to access the raw fields contained
/// within.
#[stable(feature = "metadata_ext", since = "1.1.0")]
pub trait MetadataExt {
    /// Returns the value of the `dwFileAttributes` field of this metadata.
    ///
    /// This field contains the file system attribute information for a file
    /// or directory.
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn file_attributes(&self) -> u32;

    /// Returns the value of the `ftCreationTime` field of this metadata.
    ///
    /// The returned 64-bit value represents the number of 100-nanosecond
    /// intervals since January 1, 1601 (UTC).
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn creation_time(&self) -> u64;

    /// Returns the value of the `ftLastAccessTime` field of this metadata.
    ///
    /// The returned 64-bit value represents the number of 100-nanosecond
    /// intervals since January 1, 1601 (UTC).
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn last_access_time(&self) -> u64;

    /// Returns the value of the `ftLastWriteTime` field of this metadata.
    ///
    /// The returned 64-bit value represents the number of 100-nanosecond
    /// intervals since January 1, 1601 (UTC).
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn last_write_time(&self) -> u64;

    /// Returns the value of the `nFileSize{High,Low}` fields of this
    /// metadata.
    ///
    /// The returned value does not have meaning for directories.
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn file_size(&self) -> u64;
}

#[stable(feature = "metadata_ext", since = "1.1.0")]
impl MetadataExt for Metadata {
    fn file_attributes(&self) -> u32 { self.as_inner().attrs() }
    fn creation_time(&self) -> u64 { self.as_inner().created() }
    fn last_access_time(&self) -> u64 { self.as_inner().accessed() }
    fn last_write_time(&self) -> u64 { self.as_inner().modified() }
    fn file_size(&self) -> u64 { self.as_inner().size() }
}

/// Creates a new file symbolic link on the filesystem.
///
/// The `dst` path will be a file symbolic link pointing to the `src`
/// path.
///
/// # Examples
///
/// ```ignore
/// use std::os::windows::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::symlink_file("a.txt", "b.txt"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "symlink", since = "1.1.0")]
pub fn symlink_file<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q)
                                                    -> io::Result<()> {
    sys::fs::symlink_inner(src.as_ref().as_os_str(), dst.as_ref().as_os_str(), false)
        .map_err(From::from)
}

/// Creates a new directory symlink on the filesystem.
///
/// The `dst` path will be a directory symbolic link pointing to the `src`
/// path.
///
/// # Examples
///
/// ```ignore
/// use std::os::windows::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::symlink_file("a", "b"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "symlink", since = "1.1.0")]
pub fn symlink_dir<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q)
                                                   -> io::Result<()> {
    sys::fs::symlink_inner(src.as_ref().as_os_str(), dst.as_ref().as_os_str(), true)
        .map_err(From::from)
}
