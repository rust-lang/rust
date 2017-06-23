// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows-specific extensions for the primitives in the `std::fs` module.

#![stable(feature = "rust1", since = "1.0.0")]

use fs::{self, OpenOptions, Metadata};
use io;
use path::Path;
use sys;
use sys_common::{AsInnerMut, AsInner};

/// Windows-specific extensions to [`File`].
///
/// [`File`]: ../../../fs/struct.File.html
#[stable(feature = "file_offset", since = "1.15.0")]
pub trait FileExt {
    /// Seeks to a given position and reads a number of bytes.
    ///
    /// Returns the number of bytes read.
    ///
    /// The offset is relative to the start of the file and thus independent
    /// from the current cursor. The current cursor **is** affected by this
    /// function, it is set to the end of the read.
    ///
    /// Reading beyond the end of the file will always return with a length of
    /// 0.
    ///
    /// Note that similar to `File::read`, it is not an error to return with a
    /// short read. When returning from such a short read, the file pointer is
    /// still updated.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    /// use std::fs::File;
    /// use std::os::windows::prelude::*;
    ///
    /// # fn foo() -> io::Result<()> {
    /// let mut file = File::open("foo.txt")?;
    /// let mut buffer = [0; 10];
    ///
    /// // Read 10 bytes, starting 72 bytes from the
    /// // start of the file.
    /// file.seek_read(&mut buffer[..], 72)?;
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "file_offset", since = "1.15.0")]
    fn seek_read(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;

    /// Seeks to a given position and writes a number of bytes.
    ///
    /// Returns the number of bytes written.
    ///
    /// The offset is relative to the start of the file and thus independent
    /// from the current cursor. The current cursor **is** affected by this
    /// function, it is set to the end of the write.
    ///
    /// When writing beyond the end of the file, the file is appropiately
    /// extended and the intermediate bytes are left uninitialized.
    ///
    /// Note that similar to `File::write`, it is not an error to return a
    /// short write. When returning from such a short write, the file pointer
    /// is still updated.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::os::windows::prelude::*;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut buffer = File::create("foo.txt")?;
    ///
    /// // Write a byte string starting 72 bytes from
    /// // the start of the file.
    /// buffer.seek_write(b"some bytes", 72)?;
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "file_offset", since = "1.15.0")]
    fn seek_write(&self, buf: &[u8], offset: u64) -> io::Result<usize>;
}

#[stable(feature = "file_offset", since = "1.15.0")]
impl FileExt for fs::File {
    fn seek_read(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        self.as_inner().read_at(buf, offset)
    }

    fn seek_write(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        self.as_inner().write_at(buf, offset)
    }
}

/// Windows-specific extensions to [`OpenOptions`].
///
/// [`OpenOptions`]: ../../../fs/struct.OpenOptions.html
#[stable(feature = "open_options_ext", since = "1.10.0")]
pub trait OpenOptionsExt {
    /// Overrides the `dwDesiredAccess` argument to the call to [`CreateFile`]
    /// with the specified value.
    ///
    /// This will override the `read`, `write`, and `append` flags on the
    /// `OpenOptions` structure. This method provides fine-grained control over
    /// the permissions to read, write and append data, attributes (like hidden
    /// and system), and extended attributes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    /// use std::os::windows::prelude::*;
    ///
    /// // Open without read and write permission, for example if you only need
    /// // to call `stat` on the file
    /// let file = OpenOptions::new().access_mode(0).open("foo.txt");
    /// ```
    ///
    /// [`CreateFile`]: https://msdn.microsoft.com/en-us/library/windows/desktop/aa363858.aspx
    #[stable(feature = "open_options_ext", since = "1.10.0")]
    fn access_mode(&mut self, access: u32) -> &mut Self;

    /// Overrides the `dwShareMode` argument to the call to [`CreateFile`] with
    /// the specified value.
    ///
    /// By default `share_mode` is set to
    /// `FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE`. This allows
    /// other processes to to read, write, and delete/rename the same file
    /// while it is open. Removing any of the flags will prevent other
    /// processes from performing the corresponding operation until the file
    /// handle is closed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    /// use std::os::windows::prelude::*;
    ///
    /// // Do not allow others to read or modify this file while we have it open
    /// // for writing.
    /// let file = OpenOptions::new()
    ///     .write(true)
    ///     .share_mode(0)
    ///     .open("foo.txt");
    /// ```
    ///
    /// [`CreateFile`]: https://msdn.microsoft.com/en-us/library/windows/desktop/aa363858.aspx
    #[stable(feature = "open_options_ext", since = "1.10.0")]
    fn share_mode(&mut self, val: u32) -> &mut Self;

    /// Sets extra flags for the `dwFileFlags` argument to the call to
    /// [`CreateFile2`] to the specified value (or combines it with
    /// `attributes` and `security_qos_flags` to set the `dwFlagsAndAttributes`
    /// for [`CreateFile`]).
    ///
    /// Custom flags can only set flags, not remove flags set by Rust's options.
    /// This option overwrites any previously set custom flags.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(for_demonstration_only)]
    /// extern crate winapi;
    /// # mod winapi { pub const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x04000000; }
    ///
    /// use std::fs::OpenOptions;
    /// use std::os::windows::prelude::*;
    ///
    /// let file = OpenOptions::new()
    ///     .create(true)
    ///     .write(true)
    ///     .custom_flags(winapi::FILE_FLAG_DELETE_ON_CLOSE)
    ///     .open("foo.txt");
    /// ```
    ///
    /// [`CreateFile`]: https://msdn.microsoft.com/en-us/library/windows/desktop/aa363858.aspx
    /// [`CreateFile2`]: https://msdn.microsoft.com/en-us/library/windows/desktop/hh449422.aspx
    #[stable(feature = "open_options_ext", since = "1.10.0")]
    fn custom_flags(&mut self, flags: u32) -> &mut Self;

    /// Sets the `dwFileAttributes` argument to the call to [`CreateFile2`] to
    /// the specified value (or combines it with `custom_flags` and
    /// `security_qos_flags` to set the `dwFlagsAndAttributes` for
    /// [`CreateFile`]).
    ///
    /// If a _new_ file is created because it does not yet exist and
    /// `.create(true)` or `.create_new(true)` are specified, the new file is
    /// given the attributes declared with `.attributes()`.
    ///
    /// If an _existing_ file is opened with `.create(true).truncate(true)`, its
    /// existing attributes are preserved and combined with the ones declared
    /// with `.attributes()`.
    ///
    /// In all other cases the attributes get ignored.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(for_demonstration_only)]
    /// extern crate winapi;
    /// # mod winapi { pub const FILE_ATTRIBUTE_HIDDEN: u32 = 2; }
    ///
    /// use std::fs::OpenOptions;
    /// use std::os::windows::prelude::*;
    ///
    /// let file = OpenOptions::new()
    ///     .write(true)
    ///     .create(true)
    ///     .attributes(winapi::FILE_ATTRIBUTE_HIDDEN)
    ///     .open("foo.txt");
    /// ```
    ///
    /// [`CreateFile`]: https://msdn.microsoft.com/en-us/library/windows/desktop/aa363858.aspx
    /// [`CreateFile2`]: https://msdn.microsoft.com/en-us/library/windows/desktop/hh449422.aspx
    #[stable(feature = "open_options_ext", since = "1.10.0")]
    fn attributes(&mut self, val: u32) -> &mut Self;

    /// Sets the `dwSecurityQosFlags` argument to the call to [`CreateFile2`] to
    /// the specified value (or combines it with `custom_flags` and `attributes`
    /// to set the `dwFlagsAndAttributes` for [`CreateFile`]).
    ///
    /// By default, `security_qos_flags` is set to `SECURITY_ANONYMOUS`. For
    /// information about possible values, see [Impersonation Levels] on the
    /// Windows Dev Center site.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    /// use std::os::windows::prelude::*;
    ///
    /// let file = OpenOptions::new()
    ///     .write(true)
    ///     .create(true)
    ///
    ///     // Sets the flag value to `SecurityIdentification`.
    ///     .security_qos_flags(1)
    ///
    ///     .open("foo.txt");
    /// ```
    ///
    /// [`CreateFile`]: https://msdn.microsoft.com/en-us/library/windows/desktop/aa363858.aspx
    /// [`CreateFile2`]: https://msdn.microsoft.com/en-us/library/windows/desktop/hh449422.aspx
    /// [Impersonation Levels]:
    ///     https://msdn.microsoft.com/en-us/library/windows/desktop/aa379572.aspx
    #[stable(feature = "open_options_ext", since = "1.10.0")]
    fn security_qos_flags(&mut self, flags: u32) -> &mut OpenOptions;
}

#[stable(feature = "open_options_ext", since = "1.10.0")]
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
}

/// Extension methods for [`fs::Metadata`] to access the raw fields contained
/// within.
///
/// The data members that this trait exposes correspond to the members
/// of the [`BY_HANDLE_FILE_INFORMATION`] structure.
///
/// [`fs::Metadata`]: ../../../fs/struct.Metadata.html
/// [`BY_HANDLE_FILE_INFORMATION`]:
///     https://msdn.microsoft.com/en-us/library/windows/desktop/aa363788.aspx
#[stable(feature = "metadata_ext", since = "1.1.0")]
pub trait MetadataExt {
    /// Returns the value of the `dwFileAttributes` field of this metadata.
    ///
    /// This field contains the file system attribute information for a file
    /// or directory. For possible values and their descriptions, see
    /// [File Attribute Constants] in the Windows Dev Center.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    /// use std::fs;
    /// use std::os::windows::prelude::*;
    ///
    /// # fn foo() -> io::Result<()> {
    /// let metadata = fs::metadata("foo.txt")?;
    /// let attributes = metadata.file_attributes();
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [File Attribute Constants]:
    ///     https://msdn.microsoft.com/en-us/library/windows/desktop/gg258117.aspx
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn file_attributes(&self) -> u32;

    /// Returns the value of the `ftCreationTime` field of this metadata.
    ///
    /// The returned 64-bit value is equivalent to a [`FILETIME`] struct,
    /// which represents the number of 100-nanosecond intervals since
    /// January 1, 1601 (UTC). The struct is automatically
    /// converted to a `u64` value, as that is the recommended way
    /// to use it.
    ///
    /// If the underlying filesystem does not support creation time, the
    /// returned value is 0.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    /// use std::fs;
    /// use std::os::windows::prelude::*;
    ///
    /// # fn foo() -> io::Result<()> {
    /// let metadata = fs::metadata("foo.txt")?;
    /// let creation_time = metadata.creation_time();
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`FILETIME`]: https://msdn.microsoft.com/en-us/library/windows/desktop/ms724284.aspx
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn creation_time(&self) -> u64;

    /// Returns the value of the `ftLastAccessTime` field of this metadata.
    ///
    /// The returned 64-bit value is equivalent to a [`FILETIME`] struct,
    /// which represents the number of 100-nanosecond intervals since
    /// January 1, 1601 (UTC). The struct is automatically
    /// converted to a `u64` value, as that is the recommended way
    /// to use it.
    ///
    /// For a file, the value specifies the last time that a file was read
    /// from or written to. For a directory, the value specifies when
    /// the directory was created. For both files and directories, the
    /// specified date is correct, but the time of day is always set to
    /// midnight.
    ///
    /// If the underlying filesystem does not support last access time, the
    /// returned value is 0.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    /// use std::fs;
    /// use std::os::windows::prelude::*;
    ///
    /// # fn foo() -> io::Result<()> {
    /// let metadata = fs::metadata("foo.txt")?;
    /// let last_access_time = metadata.last_access_time();
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`FILETIME`]: https://msdn.microsoft.com/en-us/library/windows/desktop/ms724284.aspx
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn last_access_time(&self) -> u64;

    /// Returns the value of the `ftLastWriteTime` field of this metadata.
    ///
    /// The returned 64-bit value is equivalent to a [`FILETIME`] struct,
    /// which represents the number of 100-nanosecond intervals since
    /// January 1, 1601 (UTC). The struct is automatically
    /// converted to a `u64` value, as that is the recommended way
    /// to use it.
    ///
    /// For a file, the value specifies the last time that a file was written
    /// to. For a directory, the structure specifies when the directory was
    /// created.
    ///
    /// If the underlying filesystem does not support the last write time
    /// time, the returned value is 0.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    /// use std::fs;
    /// use std::os::windows::prelude::*;
    ///
    /// # fn foo() -> io::Result<()> {
    /// let metadata = fs::metadata("foo.txt")?;
    /// let last_write_time = metadata.last_write_time();
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`FILETIME`]: https://msdn.microsoft.com/en-us/library/windows/desktop/ms724284.aspx
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn last_write_time(&self) -> u64;

    /// Returns the value of the `nFileSize{High,Low}` fields of this
    /// metadata.
    ///
    /// The returned value does not have meaning for directories.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    /// use std::fs;
    /// use std::os::windows::prelude::*;
    ///
    /// # fn foo() -> io::Result<()> {
    /// let metadata = fs::metadata("foo.txt")?;
    /// let file_size = metadata.file_size();
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    fn file_size(&self) -> u64;
}

#[stable(feature = "metadata_ext", since = "1.1.0")]
impl MetadataExt for Metadata {
    fn file_attributes(&self) -> u32 { self.as_inner().attrs() }
    fn creation_time(&self) -> u64 { self.as_inner().created_u64() }
    fn last_access_time(&self) -> u64 { self.as_inner().accessed_u64() }
    fn last_write_time(&self) -> u64 { self.as_inner().modified_u64() }
    fn file_size(&self) -> u64 { self.as_inner().size() }
}

/// Creates a new file symbolic link on the filesystem.
///
/// The `dst` path will be a file symbolic link pointing to the `src`
/// path.
///
/// # Examples
///
/// ```no_run
/// use std::os::windows::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// fs::symlink_file("a.txt", "b.txt")?;
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
/// ```no_run
/// use std::os::windows::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// fs::symlink_file("a", "b")?;
/// # Ok(())
/// # }
/// ```
#[stable(feature = "symlink", since = "1.1.0")]
pub fn symlink_dir<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q)
                                                   -> io::Result<()> {
    sys::fs::symlink_inner(src.as_ref(), dst.as_ref(), true)
}
