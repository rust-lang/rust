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
use sys_common::{AsInnerMut, AsInner};

/// Windows-specific extensions to `OpenOptions`
#[unstable(feature = "open_options_ext",
           reason = "may require more thought/methods",
           issue = "27720")]
pub trait OpenOptionsExt {
    /// Overrides the `dwDesiredAccess` argument to the call to `CreateFile`
    /// with the specified value.
    ///
    /// This will override the `read`, `write`, and `append` flags on the
    /// `OpenOptions` structure. This method provides fine-grained control
    /// over the permissions to read, write and append data, attributes
    /// (like hidden and system) and extended attributes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(open_options_ext)]
    /// use std::fs::OpenOptions;
    /// use std::os::windows::fs::OpenOptionsExt;
    ///
    /// // Open without read and write permission, for example if you only need to call `stat()`
    /// // on the file
    /// let file = OpenOptions::new().access_mode(0).open("foo.txt");
    /// ```
    fn access_mode(&mut self, access: u32) -> &mut Self;

    /// Overrides the `dwShareMode` argument to the call to `CreateFile` with
    /// the specified value.
    ///
    /// By default `share_mode` is set to `FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE`.
    /// Specifying less permissions denies others to read from, write to and/or
    /// delete the file while it is open.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(open_options_ext)]
    /// use std::fs::OpenOptions;
    /// use std::os::windows::fs::OpenOptionsExt;
    ///
    /// let file = OpenOptions::new().write(true)
    ///                              .share_mode(0) // Do not allow others to read or modify
    ///                              .open("foo.txt");
    /// ```
    fn share_mode(&mut self, val: u32) -> &mut Self;

    /// Sets extra flags for the `dwFileFlags` argument to the call to `CreateFile2`
    /// (or combines it with `attributes` and `security_qos_flags` to set the
    /// `dwFlagsAndAttributes` for `CreateFile`).
    ///
    /// Custom flags can only set flags, not remove flags set by Rusts options.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// extern crate winapi;
    /// use std::fs::OpenOptions;
    /// use std::os::windows::fs::OpenOptionsExt;
    ///
    /// let options = OpenOptions::new().create(true).write(true);
    /// if cfg!(windows) { options.custom_flags(winapi::FILE_FLAG_DELETE_ON_CLOSE); }
    /// let file = options.open("foo.txt");
    /// ```
    #[unstable(feature = "expand_open_options",
               reason = "recently added",
               issue = "30014")]
    fn custom_flags(&mut self, flags: u32) -> &mut Self;

    /// Sets the `dwFileAttributes` argument to the call to `CreateFile2` to
    /// the specified value (or combines it with `custom_flags` and
    /// `security_qos_flags` to set the `dwFlagsAndAttributes` for `CreateFile`).
    ///
    /// If a _new_ file is created because it does not yet exist and `.create(true)` or
    /// `.create_new(true)` are specified, the new file is given the attributes declared
    /// with `.attributes()`.
    ///
    /// If an _existing_ file is opened with `.create(true).truncate(true)`, its
    /// existing attributes are preserved and combined with the ones declared with
    /// `.attributes()`.
    ///
    /// In all other cases the attributes get ignored.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// #![feature(open_options_ext)]
    /// extern crate winapi;
    /// use std::fs::OpenOptions;
    /// use std::os::windows::fs::OpenOptionsExt;
    ///
    /// let file = OpenOptions::new().write(true).create(true)
    ///                              .attributes(winapi::FILE_ATTRIBUTE_HIDDEN)
    ///                              .open("foo.txt");
    /// ```
    fn attributes(&mut self, val: u32) -> &mut Self;

    /// Sets the `dwSecurityQosFlags` argument to the call to `CreateFile2` to
    /// the specified value (or combines it with `custom_flags` and `attributes`
    /// to set the `dwFlagsAndAttributes` for `CreateFile`).
    fn security_qos_flags(&mut self, flags: u32) -> &mut OpenOptions;

    /// Sets the `lpSecurityAttributes` argument to the call to `CreateFile` to
    /// the specified value.
    fn security_attributes(&mut self, attrs: sys::c::LPSECURITY_ATTRIBUTES) -> &mut OpenOptions;
}

#[unstable(feature = "open_options_ext",
           reason = "may require more thought/methods",
           issue = "27720")]
impl OpenOptionsExt for OpenOptions {
    fn access_mode(&mut self, access: u32) -> &mut OpenOptions {
        self.as_inner_mut().access_mode(access); self
    }

    fn share_mode(&mut self, share: u32) -> &mut OpenOptions {
        self.as_inner_mut().share_mode(share); self
    }

    fn custom_flags(&mut self, flags: u32) -> &mut OpenOptions {
        self.as_inner_mut().custom_flags(flags); self
    }

    fn attributes(&mut self, attributes: u32) -> &mut OpenOptions {
        self.as_inner_mut().attributes(attributes); self
    }

    fn security_qos_flags(&mut self, flags: u32) -> &mut OpenOptions {
        self.as_inner_mut().security_qos_flags(flags); self
    }

    fn security_attributes(&mut self, attrs: sys::c::LPSECURITY_ATTRIBUTES) -> &mut OpenOptions {
        self.as_inner_mut().security_attributes(attrs); self
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
    sys::fs::symlink_inner(src.as_ref(), dst.as_ref(), false)
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
    sys::fs::symlink_inner(src.as_ref(), dst.as_ref(), true)
}
