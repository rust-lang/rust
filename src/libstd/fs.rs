// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Filesystem manipulation operations
//!
//! This module contains basic methods to manipulate the contents of the local
//! filesystem. All methods in this module represent cross-platform filesystem
//! operations. Extra platform-specific functionality can be found in the
//! extension traits of `std::os::$platform`.

#![stable(feature = "rust1", since = "1.0.0")]

use fmt;
use ffi::OsString;
use io::{self, SeekFrom, Seek, Read, Write};
use path::{Path, PathBuf};
use sys::fs as fs_imp;
use sys_common::io::read_to_end_uninitialized;
use sys_common::{AsInnerMut, FromInner, AsInner, IntoInner};
use vec::Vec;

/// A reference to an open file on the filesystem.
///
/// An instance of a `File` can be read and/or written depending on what options
/// it was opened with. Files also implement `Seek` to alter the logical cursor
/// that the file contains internally.
///
/// # Examples
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::fs::File;
///
/// # fn foo() -> std::io::Result<()> {
/// let mut f = try!(File::create("foo.txt"));
/// try!(f.write_all(b"Hello, world!"));
///
/// let mut f = try!(File::open("foo.txt"));
/// let mut s = String::new();
/// try!(f.read_to_string(&mut s));
/// assert_eq!(s, "Hello, world!");
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct File {
    inner: fs_imp::File,
}

/// Metadata information about a file.
///
/// This structure is returned from the `metadata` function or method and
/// represents known metadata about a file such as its permissions, size,
/// modification times, etc.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Metadata(fs_imp::FileAttr);

/// Iterator over the entries in a directory.
///
/// This iterator is returned from the `read_dir` function of this module and
/// will yield instances of `io::Result<DirEntry>`. Through a `DirEntry`
/// information like the entry's path and possibly other metadata can be
/// learned.
///
/// # Failure
///
/// This `io::Result` will be an `Err` if there's some sort of intermittent
/// IO error during iteration.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ReadDir(fs_imp::ReadDir);

/// Entries returned by the `ReadDir` iterator.
///
/// An instance of `DirEntry` represents an entry inside of a directory on the
/// filesystem. Each entry can be inspected via methods to learn about the full
/// path or possibly other metadata through per-platform extension traits.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct DirEntry(fs_imp::DirEntry);

/// An iterator that recursively walks over the contents of a directory.
#[unstable(feature = "fs_walk",
           reason = "the precise semantics and defaults for a recursive walk \
                     may change and this may end up accounting for files such \
                     as symlinks differently",
           issue = "27707")]
pub struct WalkDir {
    cur: Option<ReadDir>,
    stack: Vec<io::Result<ReadDir>>,
}

/// Options and flags which can be used to configure how a file is opened.
///
/// This builder exposes the ability to configure how a `File` is opened and
/// what operations are permitted on the open file. The `File::open` and
/// `File::create` methods are aliases for commonly used options using this
/// builder.
///
/// Generally speaking, when using `OpenOptions`, you'll first call `new()`,
/// then chain calls to methods to set each option, then call `open()`, passing
/// the path of the file you're trying to open. This will give you a
/// [`io::Result`][result] with a [`File`][file] inside that you can further
/// operate on.
///
/// [result]: ../io/type.Result.html
/// [file]: struct.File.html
///
/// # Examples
///
/// Opening a file to read:
///
/// ```no_run
/// use std::fs::OpenOptions;
///
/// let file = OpenOptions::new().read(true).open("foo.txt");
/// ```
///
/// Opening a file for both reading and writing, as well as creating it if it
/// doesn't exist:
///
/// ```no_run
/// use std::fs::OpenOptions;
///
/// let file = OpenOptions::new()
///             .read(true)
///             .write(true)
///             .create(true)
///             .open("foo.txt");
/// ```
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OpenOptions(fs_imp::OpenOptions);

/// Representation of the various permissions on a file.
///
/// This module only currently provides one bit of information, `readonly`,
/// which is exposed on all currently supported platforms. Unix-specific
/// functionality, such as mode bits, is available through the
/// `os::unix::PermissionsExt` trait.
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Permissions(fs_imp::FilePermissions);

/// An structure representing a type of file with accessors for each file type.
#[stable(feature = "file_type", since = "1.1.0")]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct FileType(fs_imp::FileType);

/// A builder used to create directories in various manners.
///
/// This builder also supports platform-specific options.
#[unstable(feature = "dir_builder", reason = "recently added API",
           issue = "27710")]
pub struct DirBuilder {
    inner: fs_imp::DirBuilder,
    recursive: bool,
}

impl File {
    /// Attempts to open a file in read-only mode.
    ///
    /// See the `OpenOptions::open` method for more details.
    ///
    /// # Errors
    ///
    /// This function will return an error if `path` does not already exist.
    /// Other errors may also be returned according to `OpenOptions::open`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = try!(File::open("foo.txt"));
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<File> {
        OpenOptions::new().read(true).open(path.as_ref())
    }

    /// Opens a file in write-only mode.
    ///
    /// This function will create a file if it does not exist,
    /// and will truncate it if it does.
    ///
    /// See the `OpenOptions::open` function for more details.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = try!(File::create("foo.txt"));
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<File> {
        OpenOptions::new().write(true).create(true).truncate(true).open(path.as_ref())
    }

    /// Attempts to sync all OS-internal metadata to disk.
    ///
    /// This function will attempt to ensure that all in-core data reaches the
    /// filesystem before returning.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::prelude::*;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = try!(File::create("foo.txt"));
    /// try!(f.write_all(b"Hello, world!"));
    ///
    /// try!(f.sync_all());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn sync_all(&self) -> io::Result<()> {
        self.inner.fsync()
    }

    /// This function is similar to `sync_all`, except that it may not
    /// synchronize file metadata to the filesystem.
    ///
    /// This is intended for use cases that must synchronize content, but don't
    /// need the metadata on disk. The goal of this method is to reduce disk
    /// operations.
    ///
    /// Note that some platforms may simply implement this in terms of
    /// `sync_all`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::prelude::*;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = try!(File::create("foo.txt"));
    /// try!(f.write_all(b"Hello, world!"));
    ///
    /// try!(f.sync_data());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn sync_data(&self) -> io::Result<()> {
        self.inner.datasync()
    }

    /// Truncates or extends the underlying file, updating the size of
    /// this file to become `size`.
    ///
    /// If the `size` is less than the current file's size, then the file will
    /// be shrunk. If it is greater than the current file's size, then the file
    /// will be extended to `size` and have all of the intermediate data filled
    /// in with 0s.
    ///
    /// # Errors
    ///
    /// This function will return an error if the file is not opened for writing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = try!(File::create("foo.txt"));
    /// try!(f.set_len(10));
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set_len(&self, size: u64) -> io::Result<()> {
        self.inner.truncate(size)
    }

    /// Queries metadata about the underlying file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = try!(File::open("foo.txt"));
    /// let metadata = try!(f.metadata());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn metadata(&self) -> io::Result<Metadata> {
        self.inner.file_attr().map(Metadata)
    }
}

impl AsInner<fs_imp::File> for File {
    fn as_inner(&self) -> &fs_imp::File { &self.inner }
}
impl FromInner<fs_imp::File> for File {
    fn from_inner(f: fs_imp::File) -> File {
        File { inner: f }
    }
}
impl IntoInner<fs_imp::File> for File {
    fn into_inner(self) -> fs_imp::File {
        self.inner
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for File {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        unsafe { read_to_end_uninitialized(self, buf) }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for File {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Seek for File {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Read for &'a File {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Write for &'a File {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Seek for &'a File {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl OpenOptions {
    /// Creates a blank net set of options ready for configuration.
    ///
    /// All options are initially set to `false`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> OpenOptions {
        OpenOptions(fs_imp::OpenOptions::new())
    }

    /// Sets the option for read access.
    ///
    /// This option, when true, will indicate that the file should be
    /// `read`-able if opened.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().read(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn read(&mut self, read: bool) -> &mut OpenOptions {
        self.0.read(read); self
    }

    /// Sets the option for write access.
    ///
    /// This option, when true, will indicate that the file should be
    /// `write`-able if opened.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().write(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write(&mut self, write: bool) -> &mut OpenOptions {
        self.0.write(write); self
    }

    /// Sets the option for the append mode.
    ///
    /// This option, when true, means that writes will append to a file instead
    /// of overwriting previous contents.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().write(true).append(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn append(&mut self, append: bool) -> &mut OpenOptions {
        self.0.append(append); self
    }

    /// Sets the option for truncating a previous file.
    ///
    /// If a file is successfully opened with this option set it will truncate
    /// the file to 0 length if it already exists.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().write(true).truncate(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, truncate: bool) -> &mut OpenOptions {
        self.0.truncate(truncate); self
    }

    /// Sets the option for creating a new file.
    ///
    /// This option indicates whether a new file will be created if the file
    /// does not yet already exist.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().create(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn create(&mut self, create: bool) -> &mut OpenOptions {
        self.0.create(create); self
    }

    /// Opens a file at `path` with the options specified by `self`.
    ///
    /// # Errors
    ///
    /// This function will return an error under a number of different
    /// circumstances, to include but not limited to:
    ///
    /// * Opening a file that does not exist with read access.
    /// * Attempting to open a file with access that the user lacks
    ///   permissions for
    /// * Filesystem-level errors (full disk, etc)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn open<P: AsRef<Path>>(&self, path: P) -> io::Result<File> {
        self._open(path.as_ref())
    }

    fn _open(&self, path: &Path) -> io::Result<File> {
        let inner = try!(fs_imp::File::open(path, &self.0));
        Ok(File { inner: inner })
    }
}

impl AsInnerMut<fs_imp::OpenOptions> for OpenOptions {
    fn as_inner_mut(&mut self) -> &mut fs_imp::OpenOptions { &mut self.0 }
}

impl Metadata {
    /// Returns the file type for this metadata.
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn file_type(&self) -> FileType {
        FileType(self.0.file_type())
    }

    /// Returns whether this metadata is for a directory.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn foo() -> std::io::Result<()> {
    /// use std::fs;
    ///
    /// let metadata = try!(fs::metadata("foo.txt"));
    ///
    /// assert!(!metadata.is_dir());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_dir(&self) -> bool { self.file_type().is_dir() }

    /// Returns whether this metadata is for a regular file.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn foo() -> std::io::Result<()> {
    /// use std::fs;
    ///
    /// let metadata = try!(fs::metadata("foo.txt"));
    ///
    /// assert!(metadata.is_file());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_file(&self) -> bool { self.file_type().is_file() }

    /// Returns the size of the file, in bytes, this metadata is for.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn foo() -> std::io::Result<()> {
    /// use std::fs;
    ///
    /// let metadata = try!(fs::metadata("foo.txt"));
    ///
    /// assert_eq!(0, metadata.len());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> u64 { self.0.size() }

    /// Returns the permissions of the file this metadata is for.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn foo() -> std::io::Result<()> {
    /// use std::fs;
    ///
    /// let metadata = try!(fs::metadata("foo.txt"));
    ///
    /// assert!(!metadata.permissions().readonly());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn permissions(&self) -> Permissions {
        Permissions(self.0.perm())
    }
}

impl AsInner<fs_imp::FileAttr> for Metadata {
    fn as_inner(&self) -> &fs_imp::FileAttr { &self.0 }
}

impl Permissions {
    /// Returns whether these permissions describe a readonly file.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = try!(File::create("foo.txt"));
    /// let metadata = try!(f.metadata());
    ///
    /// assert_eq!(false, metadata.permissions().readonly());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn readonly(&self) -> bool { self.0.readonly() }

    /// Modifies the readonly flag for this set of permissions.
    ///
    /// This operation does **not** modify the filesystem. To modify the
    /// filesystem use the `fs::set_permissions` function.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let f = try!(File::create("foo.txt"));
    /// let metadata = try!(f.metadata());
    /// let mut permissions = metadata.permissions();
    ///
    /// permissions.set_readonly(true);
    ///
    /// // filesystem doesn't change
    /// assert_eq!(false, metadata.permissions().readonly());
    ///
    /// // just this particular `permissions`.
    /// assert_eq!(true, permissions.readonly());
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set_readonly(&mut self, readonly: bool) {
        self.0.set_readonly(readonly)
    }
}

impl FileType {
    /// Test whether this file type represents a directory.
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn is_dir(&self) -> bool { self.0.is_dir() }

    /// Test whether this file type represents a regular file.
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn is_file(&self) -> bool { self.0.is_file() }

    /// Test whether this file type represents a symbolic link.
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn is_symlink(&self) -> bool { self.0.is_symlink() }
}

impl AsInner<fs_imp::FileType> for FileType {
    fn as_inner(&self) -> &fs_imp::FileType { &self.0 }
}

impl FromInner<fs_imp::FilePermissions> for Permissions {
    fn from_inner(f: fs_imp::FilePermissions) -> Permissions {
        Permissions(f)
    }
}

impl AsInner<fs_imp::FilePermissions> for Permissions {
    fn as_inner(&self) -> &fs_imp::FilePermissions { &self.0 }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        self.0.next().map(|entry| entry.map(DirEntry))
    }
}

impl DirEntry {
    /// Returns the full path to the file that this entry represents.
    ///
    /// The full path is created by joining the original path to `read_dir` or
    /// `walk_dir` with the filename of this entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs;
    /// # fn foo() -> std::io::Result<()> {
    /// for entry in try!(fs::read_dir(".")) {
    ///     let dir = try!(entry);
    ///     println!("{:?}", dir.path());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// This prints output like:
    ///
    /// ```text
    /// "./whatever.txt"
    /// "./foo.html"
    /// "./hello_world.rs"
    /// ```
    ///
    /// The exact text, of course, depends on what files you have in `.`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn path(&self) -> PathBuf { self.0.path() }

    /// Return the metadata for the file that this entry points at.
    ///
    /// This function will not traverse symlinks if this entry points at a
    /// symlink.
    ///
    /// # Platform behavior
    ///
    /// On Windows this function is cheap to call (no extra system calls
    /// needed), but on Unix platforms this function is the equivalent of
    /// calling `symlink_metadata` on the path.
    #[stable(feature = "dir_entry_ext", since = "1.1.0")]
    pub fn metadata(&self) -> io::Result<Metadata> {
        self.0.metadata().map(Metadata)
    }

    /// Return the file type for the file that this entry points at.
    ///
    /// This function will not traverse symlinks if this entry points at a
    /// symlink.
    ///
    /// # Platform behavior
    ///
    /// On Windows and most Unix platforms this function is free (no extra
    /// system calls needed), but some Unix platforms may require the equivalent
    /// call to `symlink_metadata` to learn about the target file type.
    #[stable(feature = "dir_entry_ext", since = "1.1.0")]
    pub fn file_type(&self) -> io::Result<FileType> {
        self.0.file_type().map(FileType)
    }

    /// Returns the bare file name of this directory entry without any other
    /// leading path component.
    #[stable(feature = "dir_entry_ext", since = "1.1.0")]
    pub fn file_name(&self) -> OsString {
        self.0.file_name()
    }
}

impl AsInner<fs_imp::DirEntry> for DirEntry {
    fn as_inner(&self) -> &fs_imp::DirEntry { &self.0 }
}

/// Removes a file from the filesystem.
///
/// Note that there is no
/// guarantee that the file is immediately deleted (e.g. depending on
/// platform, other open file descriptors may prevent immediate removal).
///
/// # Errors
///
/// This function will return an error if `path` points to a directory, if the
/// user lacks permissions to remove the file, or if some other filesystem-level
/// error occurs.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::remove_file("a.txt"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()> {
    fs_imp::unlink(path.as_ref())
}

/// Given a path, query the file system to get information about a file,
/// directory, etc.
///
/// This function will traverse symbolic links to query information about the
/// destination file.
///
/// # Examples
///
/// ```rust
/// # fn foo() -> std::io::Result<()> {
/// use std::fs;
///
/// let attr = try!(fs::metadata("/some/file/path.txt"));
/// // inspect attr ...
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// This function will return an error if the user lacks the requisite
/// permissions to perform a `metadata` call on the given `path` or if there
/// is no entry in the filesystem at the provided path.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn metadata<P: AsRef<Path>>(path: P) -> io::Result<Metadata> {
    fs_imp::stat(path.as_ref()).map(Metadata)
}

/// Query the metadata about a file without following symlinks.
///
/// # Examples
///
/// ```rust
/// # fn foo() -> std::io::Result<()> {
/// use std::fs;
///
/// let attr = try!(fs::symlink_metadata("/some/file/path.txt"));
/// // inspect attr ...
/// # Ok(())
/// # }
/// ```
#[stable(feature = "symlink_metadata", since = "1.1.0")]
pub fn symlink_metadata<P: AsRef<Path>>(path: P) -> io::Result<Metadata> {
    fs_imp::lstat(path.as_ref()).map(Metadata)
}

/// Rename a file or directory to a new name.
///
/// This will not work if the new name is on a different mount point.
///
/// # Errors
///
/// This function will return an error if the provided `from` doesn't exist, if
/// the process lacks permissions to view the contents, if `from` and `to`
/// reside on separate filesystems, or if some other intermittent I/O error
/// occurs.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::rename("a.txt", "b.txt"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()> {
    fs_imp::rename(from.as_ref(), to.as_ref())
}

/// Copies the contents of one file to another. This function will also
/// copy the permission bits of the original file to the destination file.
///
/// This function will **overwrite** the contents of `to`.
///
/// Note that if `from` and `to` both point to the same file, then the file
/// will likely get truncated by this operation.
///
/// On success, the total number of bytes copied is returned.
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The `from` path is not a file
/// * The `from` file does not exist
/// * The current process does not have the permission rights to access
///   `from` or write `to`
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::copy("foo.txt", "bar.txt"));
/// # Ok(()) }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
    fs_imp::copy(from.as_ref(), to.as_ref())
}

/// Creates a new hard link on the filesystem.
///
/// The `dst` path will be a link pointing to the `src` path. Note that systems
/// often require these two paths to both be located on the same filesystem.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::hard_link("a.txt", "b.txt"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn hard_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> io::Result<()> {
    fs_imp::link(src.as_ref(), dst.as_ref())
}

/// Creates a new symbolic link on the filesystem.
///
/// The `dst` path will be a symbolic link pointing to the `src` path.
/// On Windows, this will be a file symlink, not a directory symlink;
/// for this reason, the platform-specific `std::os::unix::fs::symlink`
/// and `std::os::windows::fs::{symlink_file, symlink_dir}` should be
/// used instead to make the intent explicit.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::soft_link("a.txt", "b.txt"));
/// # Ok(())
/// # }
/// ```
#[deprecated(since = "1.1.0",
             reason = "replaced with std::os::unix::fs::symlink and \
                       std::os::windows::fs::{symlink_file, symlink_dir}")]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn soft_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> io::Result<()> {
    fs_imp::symlink(src.as_ref(), dst.as_ref())
}

/// Reads a symbolic link, returning the file that the link points to.
///
/// # Errors
///
/// This function will return an error on failure. Failure conditions include
/// reading a file that does not exist or reading a file that is not a symbolic
/// link.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// let path = try!(fs::read_link("a.txt"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn read_link<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    fs_imp::readlink(path.as_ref())
}

/// Returns the canonical form of a path with all intermediate components
/// normalized and symbolic links resolved.
///
/// This function may return an error in situations like where the path does not
/// exist, a component in the path is not a directory, or an I/O error happens.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// let path = try!(fs::canonicalize("../a/../foo.txt"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "fs_canonicalize", since = "1.5.0")]
pub fn canonicalize<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    fs_imp::canonicalize(path.as_ref())
}

/// Creates a new, empty directory at the provided path
///
/// # Errors
///
/// This function will return an error if the user lacks permissions to make a
/// new directory at the provided `path`, or if the directory already exists.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::create_dir("/some/dir"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    DirBuilder::new().create(path.as_ref())
}

/// Recursively create a directory and all of its parent components if they
/// are missing.
///
/// # Errors
///
/// This function will fail if any directory in the path specified by `path`
/// does not already exist and it could not be created otherwise. The specific
/// error conditions for when a directory is being created (after it is
/// determined to not exist) are outlined by `fs::create_dir`.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::create_dir_all("/some/dir"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    DirBuilder::new().recursive(true).create(path.as_ref())
}

/// Removes an existing, empty directory.
///
/// # Errors
///
/// This function will return an error if the user lacks permissions to remove
/// the directory at the provided `path`, or if the directory isn't empty.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::remove_dir("/some/dir"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn remove_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    fs_imp::rmdir(path.as_ref())
}

/// Removes a directory at this path, after removing all its contents. Use
/// carefully!
///
/// This function does **not** follow symbolic links and it will simply remove the
/// symbolic link itself.
///
/// # Errors
///
/// See `file::remove_file` and `fs::remove_dir`.
///
/// # Examples
///
/// ```
/// use std::fs;
///
/// # fn foo() -> std::io::Result<()> {
/// try!(fs::remove_dir_all("/some/dir"));
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    _remove_dir_all(path.as_ref())
}

fn _remove_dir_all(path: &Path) -> io::Result<()> {
    for child in try!(read_dir(path)) {
        let child = try!(child).path();
        let stat = try!(symlink_metadata(&*child));
        if stat.is_dir() {
            try!(remove_dir_all(&*child));
        } else {
            try!(remove_file(&*child));
        }
    }
    remove_dir(path)
}

/// Returns an iterator over the entries within a directory.
///
/// The iterator will yield instances of `io::Result<DirEntry>`. New errors may
/// be encountered after an iterator is initially constructed.
///
/// # Examples
///
/// ```
/// use std::io;
/// use std::fs::{self, DirEntry};
/// use std::path::Path;
///
/// // one possible implementation of fs::walk_dir only visiting files
/// fn visit_dirs(dir: &Path, cb: &Fn(&DirEntry)) -> io::Result<()> {
///     if try!(fs::metadata(dir)).is_dir() {
///         for entry in try!(fs::read_dir(dir)) {
///             let entry = try!(entry);
///             if try!(fs::metadata(entry.path())).is_dir() {
///                 try!(visit_dirs(&entry.path(), cb));
///             } else {
///                 cb(&entry);
///             }
///         }
///     }
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// This function will return an error if the provided `path` doesn't exist, if
/// the process lacks permissions to view the contents or if the `path` points
/// at a non-directory file
#[stable(feature = "rust1", since = "1.0.0")]
pub fn read_dir<P: AsRef<Path>>(path: P) -> io::Result<ReadDir> {
    fs_imp::readdir(path.as_ref()).map(ReadDir)
}

/// Returns an iterator that will recursively walk the directory structure
/// rooted at `path`.
///
/// The path given will not be iterated over, and this will perform iteration in
/// some top-down order.  The contents of unreadable subdirectories are ignored.
///
/// The iterator will yield instances of `io::Result<DirEntry>`. New errors may
/// be encountered after an iterator is initially constructed.
#[unstable(feature = "fs_walk",
           reason = "the precise semantics and defaults for a recursive walk \
                     may change and this may end up accounting for files such \
                     as symlinks differently",
           issue = "27707")]
pub fn walk_dir<P: AsRef<Path>>(path: P) -> io::Result<WalkDir> {
    _walk_dir(path.as_ref())
}

fn _walk_dir(path: &Path) -> io::Result<WalkDir> {
    let start = try!(read_dir(path));
    Ok(WalkDir { cur: Some(start), stack: Vec::new() })
}

#[unstable(feature = "fs_walk", issue = "27707")]
impl Iterator for WalkDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        loop {
            if let Some(ref mut cur) = self.cur {
                match cur.next() {
                    Some(Err(e)) => return Some(Err(e)),
                    Some(Ok(next)) => {
                        let path = next.path();
                        if path.is_dir() {
                            self.stack.push(read_dir(&*path));
                        }
                        return Some(Ok(next))
                    }
                    None => {}
                }
            }
            self.cur = None;
            match self.stack.pop() {
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(next)) => self.cur = Some(next),
                None => return None,
            }
        }
    }
}

/// Utility methods for paths.
#[unstable(feature = "path_ext_deprecated",
           reason = "The precise set of methods exposed on this trait may \
                     change and some methods may be removed.  For stable code, \
                     see the std::fs::metadata function.",
           issue = "27725")]
#[deprecated(since = "1.5.0", reason = "replaced with inherent methods")]
pub trait PathExt {
    /// Gets information on the file, directory, etc at this path.
    ///
    /// Consult the `fs::metadata` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with
    /// `fs::metadata`.
    fn metadata(&self) -> io::Result<Metadata>;

    /// Gets information on the file, directory, etc at this path.
    ///
    /// Consult the `fs::symlink_metadata` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with
    /// `fs::symlink_metadata`.
    fn symlink_metadata(&self) -> io::Result<Metadata>;

    /// Returns the canonical form of a path, normalizing all components and
    /// eliminate all symlinks.
    ///
    /// This call preserves identical runtime/error semantics with
    /// `fs::canonicalize`.
    fn canonicalize(&self) -> io::Result<PathBuf>;

    /// Reads the symlink at this path.
    ///
    /// For more information see `fs::read_link`.
    fn read_link(&self) -> io::Result<PathBuf>;

    /// Reads the directory at this path.
    ///
    /// For more information see `fs::read_dir`.
    fn read_dir(&self) -> io::Result<ReadDir>;

    /// Boolean value indicator whether the underlying file exists on the local
    /// filesystem. Returns false in exactly the cases where `fs::stat` fails.
    fn exists(&self) -> bool;

    /// Whether the underlying implementation (be it a file path, or something
    /// else) points at a "regular file" on the FS. Will return false for paths
    /// to non-existent locations or directories or other non-regular files
    /// (named pipes, etc). Follows links when making this determination.
    fn is_file(&self) -> bool;

    /// Whether the underlying implementation (be it a file path, or something
    /// else) is pointing at a directory in the underlying FS. Will return
    /// false for paths to non-existent locations or if the item is not a
    /// directory (eg files, named pipes, etc). Follows links when making this
    /// determination.
    fn is_dir(&self) -> bool;
}

#[allow(deprecated)]
impl PathExt for Path {
    fn metadata(&self) -> io::Result<Metadata> { metadata(self) }
    fn symlink_metadata(&self) -> io::Result<Metadata> { symlink_metadata(self) }
    fn canonicalize(&self) -> io::Result<PathBuf> { canonicalize(self) }
    fn read_link(&self) -> io::Result<PathBuf> { read_link(self) }
    fn read_dir(&self) -> io::Result<ReadDir> { read_dir(self) }
    fn exists(&self) -> bool { metadata(self).is_ok() }

    fn is_file(&self) -> bool {
        metadata(self).map(|s| s.is_file()).unwrap_or(false)
    }

    fn is_dir(&self) -> bool {
        metadata(self).map(|s| s.is_dir()).unwrap_or(false)
    }
}

/// Changes the permissions found on a file or a directory.
///
/// # Examples
///
/// ```
/// # fn foo() -> std::io::Result<()> {
/// use std::fs;
///
/// let mut perms = try!(fs::metadata("foo.txt")).permissions();
/// perms.set_readonly(true);
/// try!(fs::set_permissions("foo.txt", perms));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// This function will return an error if the provided `path` doesn't exist, if
/// the process lacks permissions to change the attributes of the file, or if
/// some other I/O error is encountered.
#[stable(feature = "set_permissions", since = "1.1.0")]
pub fn set_permissions<P: AsRef<Path>>(path: P, perm: Permissions)
                                       -> io::Result<()> {
    fs_imp::set_perm(path.as_ref(), perm.0)
}

#[unstable(feature = "dir_builder", reason = "recently added API",
           issue = "27710")]
impl DirBuilder {
    /// Creates a new set of options with default mode/security settings for all
    /// platforms and also non-recursive.
    pub fn new() -> DirBuilder {
        DirBuilder {
            inner: fs_imp::DirBuilder::new(),
            recursive: false,
        }
    }

    /// Indicate that directories create should be created recursively, creating
    /// all parent directories if they do not exist with the same security and
    /// permissions settings.
    ///
    /// This option defaults to `false`
    pub fn recursive(&mut self, recursive: bool) -> &mut Self {
        self.recursive = recursive;
        self
    }

    /// Create the specified directory with the options configured in this
    /// builder.
    pub fn create<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        self._create(path.as_ref())
    }

    fn _create(&self, path: &Path) -> io::Result<()> {
        if self.recursive {
            self.create_dir_all(path)
        } else {
            self.inner.mkdir(path)
        }
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        if path == Path::new("") || path.is_dir() { return Ok(()) }
        if let Some(p) = path.parent() {
            try!(self.create_dir_all(p))
        }
        self.inner.mkdir(path)
    }
}

impl AsInnerMut<fs_imp::DirBuilder> for DirBuilder {
    fn as_inner_mut(&mut self) -> &mut fs_imp::DirBuilder {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)] //rand

    use prelude::v1::*;
    use io::prelude::*;

    use env;
    use fs::{self, File, OpenOptions};
    use io::{ErrorKind, SeekFrom};
    use path::PathBuf;
    use path::Path as Path2;
    use os;
    use rand::{self, StdRng, Rng};
    use str;

    macro_rules! check { ($e:expr) => (
        match $e {
            Ok(t) => t,
            Err(e) => panic!("{} failed with: {}", stringify!($e), e),
        }
    ) }

    macro_rules! error { ($e:expr, $s:expr) => (
        match $e {
            Ok(_) => panic!("Unexpected success. Should've been: {:?}", $s),
            Err(ref err) => assert!(err.to_string().contains($s),
                                    format!("`{}` did not contain `{}`", err, $s))
        }
    ) }

    pub struct TempDir(PathBuf);

    impl TempDir {
        fn join(&self, path: &str) -> PathBuf {
            let TempDir(ref p) = *self;
            p.join(path)
        }

        fn path<'a>(&'a self) -> &'a Path2 {
            let TempDir(ref p) = *self;
            p
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Gee, seeing how we're testing the fs module I sure hope that we
            // at least implement this correctly!
            let TempDir(ref p) = *self;
            check!(fs::remove_dir_all(p));
        }
    }

    pub fn tmpdir() -> TempDir {
        let p = env::temp_dir();
        let mut r = rand::thread_rng();
        let ret = p.join(&format!("rust-{}", r.next_u32()));
        check!(fs::create_dir(&ret));
        TempDir(ret)
    }

    #[test]
    fn file_test_io_smoke_test() {
        let message = "it's alright. have a good time";
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test.txt");
        {
            let mut write_stream = check!(File::create(filename));
            check!(write_stream.write(message.as_bytes()));
        }
        {
            let mut read_stream = check!(File::open(filename));
            let mut read_buf = [0; 1028];
            let read_str = match check!(read_stream.read(&mut read_buf)) {
                0 => panic!("shouldn't happen"),
                n => str::from_utf8(&read_buf[..n]).unwrap().to_string()
            };
            assert_eq!(read_str, message);
        }
        check!(fs::remove_file(filename));
    }

    #[test]
    fn invalid_path_raises() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_that_does_not_exist.txt");
        let result = File::open(filename);

        if cfg!(unix) {
            error!(result, "o such file or directory");
        }
        // error!(result, "couldn't open path as file");
        // error!(result, format!("path={}; mode=open; access=read", filename.display()));
    }

    #[test]
    fn file_test_iounlinking_invalid_path_should_raise_condition() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_another_file_that_does_not_exist.txt");

        let result = fs::remove_file(filename);

        if cfg!(unix) {
            error!(result, "o such file or directory");
        }
        // error!(result, "couldn't unlink path");
        // error!(result, format!("path={}", filename.display()));
    }

    #[test]
    fn file_test_io_non_positional_read() {
        let message: &str = "ten-four";
        let mut read_mem = [0; 8];
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_positional.txt");
        {
            let mut rw_stream = check!(File::create(filename));
            check!(rw_stream.write(message.as_bytes()));
        }
        {
            let mut read_stream = check!(File::open(filename));
            {
                let read_buf = &mut read_mem[0..4];
                check!(read_stream.read(read_buf));
            }
            {
                let read_buf = &mut read_mem[4..8];
                check!(read_stream.read(read_buf));
            }
        }
        check!(fs::remove_file(filename));
        let read_str = str::from_utf8(&read_mem).unwrap();
        assert_eq!(read_str, message);
    }

    #[test]
    fn file_test_io_seek_and_tell_smoke_test() {
        let message = "ten-four";
        let mut read_mem = [0; 4];
        let set_cursor = 4 as u64;
        let mut tell_pos_pre_read;
        let mut tell_pos_post_read;
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_seeking.txt");
        {
            let mut rw_stream = check!(File::create(filename));
            check!(rw_stream.write(message.as_bytes()));
        }
        {
            let mut read_stream = check!(File::open(filename));
            check!(read_stream.seek(SeekFrom::Start(set_cursor)));
            tell_pos_pre_read = check!(read_stream.seek(SeekFrom::Current(0)));
            check!(read_stream.read(&mut read_mem));
            tell_pos_post_read = check!(read_stream.seek(SeekFrom::Current(0)));
        }
        check!(fs::remove_file(filename));
        let read_str = str::from_utf8(&read_mem).unwrap();
        assert_eq!(read_str, &message[4..8]);
        assert_eq!(tell_pos_pre_read, set_cursor);
        assert_eq!(tell_pos_post_read, message.len() as u64);
    }

    #[test]
    fn file_test_io_seek_and_write() {
        let initial_msg =   "food-is-yummy";
        let overwrite_msg =    "-the-bar!!";
        let final_msg =     "foo-the-bar!!";
        let seek_idx = 3;
        let mut read_mem = [0; 13];
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_seek_and_write.txt");
        {
            let mut rw_stream = check!(File::create(filename));
            check!(rw_stream.write(initial_msg.as_bytes()));
            check!(rw_stream.seek(SeekFrom::Start(seek_idx)));
            check!(rw_stream.write(overwrite_msg.as_bytes()));
        }
        {
            let mut read_stream = check!(File::open(filename));
            check!(read_stream.read(&mut read_mem));
        }
        check!(fs::remove_file(filename));
        let read_str = str::from_utf8(&read_mem).unwrap();
        assert!(read_str == final_msg);
    }

    #[test]
    fn file_test_io_seek_shakedown() {
        //                   01234567890123
        let initial_msg =   "qwer-asdf-zxcv";
        let chunk_one: &str = "qwer";
        let chunk_two: &str = "asdf";
        let chunk_three: &str = "zxcv";
        let mut read_mem = [0; 4];
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_seek_shakedown.txt");
        {
            let mut rw_stream = check!(File::create(filename));
            check!(rw_stream.write(initial_msg.as_bytes()));
        }
        {
            let mut read_stream = check!(File::open(filename));

            check!(read_stream.seek(SeekFrom::End(-4)));
            check!(read_stream.read(&mut read_mem));
            assert_eq!(str::from_utf8(&read_mem).unwrap(), chunk_three);

            check!(read_stream.seek(SeekFrom::Current(-9)));
            check!(read_stream.read(&mut read_mem));
            assert_eq!(str::from_utf8(&read_mem).unwrap(), chunk_two);

            check!(read_stream.seek(SeekFrom::Start(0)));
            check!(read_stream.read(&mut read_mem));
            assert_eq!(str::from_utf8(&read_mem).unwrap(), chunk_one);
        }
        check!(fs::remove_file(filename));
    }

    #[test]
    fn file_test_stat_is_correct_on_is_file() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_stat_correct_on_is_file.txt");
        {
            let mut opts = OpenOptions::new();
            let mut fs = check!(opts.read(true).write(true)
                                    .create(true).open(filename));
            let msg = "hw";
            fs.write(msg.as_bytes()).unwrap();

            let fstat_res = check!(fs.metadata());
            assert!(fstat_res.is_file());
        }
        let stat_res_fn = check!(fs::metadata(filename));
        assert!(stat_res_fn.is_file());
        let stat_res_meth = check!(filename.metadata());
        assert!(stat_res_meth.is_file());
        check!(fs::remove_file(filename));
    }

    #[test]
    fn file_test_stat_is_correct_on_is_dir() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_stat_correct_on_is_dir");
        check!(fs::create_dir(filename));
        let stat_res_fn = check!(fs::metadata(filename));
        assert!(stat_res_fn.is_dir());
        let stat_res_meth = check!(filename.metadata());
        assert!(stat_res_meth.is_dir());
        check!(fs::remove_dir(filename));
    }

    #[test]
    fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("fileinfo_false_on_dir");
        check!(fs::create_dir(dir));
        assert!(dir.is_file() == false);
        check!(fs::remove_dir(dir));
    }

    #[test]
    fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
        let tmpdir = tmpdir();
        let file = &tmpdir.join("fileinfo_check_exists_b_and_a.txt");
        check!(check!(File::create(file)).write(b"foo"));
        assert!(file.exists());
        check!(fs::remove_file(file));
        assert!(!file.exists());
    }

    #[test]
    fn file_test_directoryinfo_check_exists_before_and_after_mkdir() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("before_and_after_dir");
        assert!(!dir.exists());
        check!(fs::create_dir(dir));
        assert!(dir.exists());
        assert!(dir.is_dir());
        check!(fs::remove_dir(dir));
        assert!(!dir.exists());
    }

    #[test]
    fn file_test_directoryinfo_readdir() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("di_readdir");
        check!(fs::create_dir(dir));
        let prefix = "foo";
        for n in 0..3 {
            let f = dir.join(&format!("{}.txt", n));
            let mut w = check!(File::create(&f));
            let msg_str = format!("{}{}", prefix, n.to_string());
            let msg = msg_str.as_bytes();
            check!(w.write(msg));
        }
        let files = check!(fs::read_dir(dir));
        let mut mem = [0; 4];
        for f in files {
            let f = f.unwrap().path();
            {
                let n = f.file_stem().unwrap();
                check!(check!(File::open(&f)).read(&mut mem));
                let read_str = str::from_utf8(&mem).unwrap();
                let expected = format!("{}{}", prefix, n.to_str().unwrap());
                assert_eq!(expected, read_str);
            }
            check!(fs::remove_file(&f));
        }
        check!(fs::remove_dir(dir));
    }

    #[test]
    fn file_test_walk_dir() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("walk_dir");
        check!(fs::create_dir(dir));

        let dir1 = &dir.join("01/02/03");
        check!(fs::create_dir_all(dir1));
        check!(File::create(&dir1.join("04")));

        let dir2 = &dir.join("11/12/13");
        check!(fs::create_dir_all(dir2));
        check!(File::create(&dir2.join("14")));

        let files = check!(fs::walk_dir(dir));
        let mut cur = [0; 2];
        for f in files {
            let f = f.unwrap().path();
            let stem = f.file_stem().unwrap().to_str().unwrap();
            let root = stem.as_bytes()[0] - b'0';
            let name = stem.as_bytes()[1] - b'0';
            assert!(cur[root as usize] < name);
            cur[root as usize] = name;
        }

        check!(fs::remove_dir_all(dir));
    }

    #[test]
    fn mkdir_path_already_exists_error() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("mkdir_error_twice");
        check!(fs::create_dir(dir));
        let e = fs::create_dir(dir).err().unwrap();
        assert_eq!(e.kind(), ErrorKind::AlreadyExists);
    }

    #[test]
    fn recursive_mkdir() {
        let tmpdir = tmpdir();
        let dir = tmpdir.join("d1/d2");
        check!(fs::create_dir_all(&dir));
        assert!(dir.is_dir())
    }

    #[test]
    fn recursive_mkdir_failure() {
        let tmpdir = tmpdir();
        let dir = tmpdir.join("d1");
        let file = dir.join("f1");

        check!(fs::create_dir_all(&dir));
        check!(File::create(&file));

        let result = fs::create_dir_all(&file);

        assert!(result.is_err());
        // error!(result, "couldn't recursively mkdir");
        // error!(result, "couldn't create directory");
        // error!(result, "mode=0700");
        // error!(result, format!("path={}", file.display()));
    }

    #[test]
    fn recursive_mkdir_slash() {
        check!(fs::create_dir_all(&Path2::new("/")));
    }

    // FIXME(#12795) depends on lstat to work on windows
    #[cfg(not(windows))]
    #[test]
    fn recursive_rmdir() {
        let tmpdir = tmpdir();
        let d1 = tmpdir.join("d1");
        let dt = d1.join("t");
        let dtt = dt.join("t");
        let d2 = tmpdir.join("d2");
        let canary = d2.join("do_not_delete");
        check!(fs::create_dir_all(&dtt));
        check!(fs::create_dir_all(&d2));
        check!(check!(File::create(&canary)).write(b"foo"));
        check!(fs::soft_link(&d2, &dt.join("d2")));
        check!(fs::remove_dir_all(&d1));

        assert!(!d1.is_dir());
        assert!(canary.exists());
    }

    #[test]
    fn unicode_path_is_dir() {
        assert!(Path2::new(".").is_dir());
        assert!(!Path2::new("test/stdtest/fs.rs").is_dir());

        let tmpdir = tmpdir();

        let mut dirpath = tmpdir.path().to_path_buf();
        dirpath.push("test-가一ー你好");
        check!(fs::create_dir(&dirpath));
        assert!(dirpath.is_dir());

        let mut filepath = dirpath;
        filepath.push("unicode-file-\u{ac00}\u{4e00}\u{30fc}\u{4f60}\u{597d}.rs");
        check!(File::create(&filepath)); // ignore return; touch only
        assert!(!filepath.is_dir());
        assert!(filepath.exists());
    }

    #[test]
    fn unicode_path_exists() {
        assert!(Path2::new(".").exists());
        assert!(!Path2::new("test/nonexistent-bogus-path").exists());

        let tmpdir = tmpdir();
        let unicode = tmpdir.path();
        let unicode = unicode.join(&format!("test-각丁ー再见"));
        check!(fs::create_dir(&unicode));
        assert!(unicode.exists());
        assert!(!Path2::new("test/unicode-bogus-path-각丁ー再见").exists());
    }

    #[test]
    fn copy_file_does_not_exist() {
        let from = Path2::new("test/nonexistent-bogus-path");
        let to = Path2::new("test/other-bogus-path");

        match fs::copy(&from, &to) {
            Ok(..) => panic!(),
            Err(..) => {
                assert!(!from.exists());
                assert!(!to.exists());
            }
        }
    }

    #[test]
    fn copy_src_does_not_exist() {
        let tmpdir = tmpdir();
        let from = Path2::new("test/nonexistent-bogus-path");
        let to = tmpdir.join("out.txt");
        check!(check!(File::create(&to)).write(b"hello"));
        assert!(fs::copy(&from, &to).is_err());
        assert!(!from.exists());
        let mut v = Vec::new();
        check!(check!(File::open(&to)).read_to_end(&mut v));
        assert_eq!(v, b"hello");
    }

    #[test]
    fn copy_file_ok() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        check!(check!(File::create(&input)).write(b"hello"));
        check!(fs::copy(&input, &out));
        let mut v = Vec::new();
        check!(check!(File::open(&out)).read_to_end(&mut v));
        assert_eq!(v, b"hello");

        assert_eq!(check!(input.metadata()).permissions(),
                   check!(out.metadata()).permissions());
    }

    #[test]
    fn copy_file_dst_dir() {
        let tmpdir = tmpdir();
        let out = tmpdir.join("out");

        check!(File::create(&out));
        match fs::copy(&*out, tmpdir.path()) {
            Ok(..) => panic!(), Err(..) => {}
        }
    }

    #[test]
    fn copy_file_dst_exists() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in");
        let output = tmpdir.join("out");

        check!(check!(File::create(&input)).write("foo".as_bytes()));
        check!(check!(File::create(&output)).write("bar".as_bytes()));
        check!(fs::copy(&input, &output));

        let mut v = Vec::new();
        check!(check!(File::open(&output)).read_to_end(&mut v));
        assert_eq!(v, b"foo".to_vec());
    }

    #[test]
    fn copy_file_src_dir() {
        let tmpdir = tmpdir();
        let out = tmpdir.join("out");

        match fs::copy(tmpdir.path(), &out) {
            Ok(..) => panic!(), Err(..) => {}
        }
        assert!(!out.exists());
    }

    #[test]
    fn copy_file_preserves_perm_bits() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        let attr = check!(check!(File::create(&input)).metadata());
        let mut p = attr.permissions();
        p.set_readonly(true);
        check!(fs::set_permissions(&input, p));
        check!(fs::copy(&input, &out));
        assert!(check!(out.metadata()).permissions().readonly());
        check!(fs::set_permissions(&input, attr.permissions()));
        check!(fs::set_permissions(&out, attr.permissions()));
    }

    #[cfg(windows)]
    #[test]
    fn copy_file_preserves_streams() {
        let tmp = tmpdir();
        check!(check!(File::create(tmp.join("in.txt:bunny"))).write("carrot".as_bytes()));
        assert_eq!(check!(fs::copy(tmp.join("in.txt"), tmp.join("out.txt"))), 6);
        assert_eq!(check!(tmp.join("out.txt").metadata()).len(), 0);
        let mut v = Vec::new();
        check!(check!(File::open(tmp.join("out.txt:bunny"))).read_to_end(&mut v));
        assert_eq!(v, b"carrot".to_vec());
    }

    #[cfg(not(windows))] // FIXME(#10264) operation not permitted?
    #[test]
    fn symlinks_work() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        check!(check!(File::create(&input)).write("foobar".as_bytes()));
        check!(fs::soft_link(&input, &out));
        // if cfg!(not(windows)) {
        //     assert_eq!(check!(lstat(&out)).kind, FileType::Symlink);
        //     assert_eq!(check!(out.lstat()).kind, FileType::Symlink);
        // }
        assert_eq!(check!(fs::metadata(&out)).len(),
                   check!(fs::metadata(&input)).len());
        let mut v = Vec::new();
        check!(check!(File::open(&out)).read_to_end(&mut v));
        assert_eq!(v, b"foobar".to_vec());
    }

    #[cfg(not(windows))] // apparently windows doesn't like symlinks
    #[test]
    fn symlink_noexist() {
        let tmpdir = tmpdir();
        // symlinks can point to things that don't exist
        check!(fs::soft_link(&tmpdir.join("foo"), &tmpdir.join("bar")));
        assert_eq!(check!(fs::read_link(&tmpdir.join("bar"))),
                   tmpdir.join("foo"));
    }

    #[test]
    fn readlink_not_symlink() {
        let tmpdir = tmpdir();
        match fs::read_link(tmpdir.path()) {
            Ok(..) => panic!("wanted a failure"),
            Err(..) => {}
        }
    }

    #[test]
    fn links_work() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        check!(check!(File::create(&input)).write("foobar".as_bytes()));
        check!(fs::hard_link(&input, &out));
        assert_eq!(check!(fs::metadata(&out)).len(),
                   check!(fs::metadata(&input)).len());
        assert_eq!(check!(fs::metadata(&out)).len(),
                   check!(input.metadata()).len());
        let mut v = Vec::new();
        check!(check!(File::open(&out)).read_to_end(&mut v));
        assert_eq!(v, b"foobar".to_vec());

        // can't link to yourself
        match fs::hard_link(&input, &input) {
            Ok(..) => panic!("wanted a failure"),
            Err(..) => {}
        }
        // can't link to something that doesn't exist
        match fs::hard_link(&tmpdir.join("foo"), &tmpdir.join("bar")) {
            Ok(..) => panic!("wanted a failure"),
            Err(..) => {}
        }
    }

    #[test]
    fn chmod_works() {
        let tmpdir = tmpdir();
        let file = tmpdir.join("in.txt");

        check!(File::create(&file));
        let attr = check!(fs::metadata(&file));
        assert!(!attr.permissions().readonly());
        let mut p = attr.permissions();
        p.set_readonly(true);
        check!(fs::set_permissions(&file, p.clone()));
        let attr = check!(fs::metadata(&file));
        assert!(attr.permissions().readonly());

        match fs::set_permissions(&tmpdir.join("foo"), p.clone()) {
            Ok(..) => panic!("wanted an error"),
            Err(..) => {}
        }

        p.set_readonly(false);
        check!(fs::set_permissions(&file, p));
    }

    #[test]
    fn sync_doesnt_kill_anything() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let mut file = check!(File::create(&path));
        check!(file.sync_all());
        check!(file.sync_data());
        check!(file.write(b"foo"));
        check!(file.sync_all());
        check!(file.sync_data());
    }

    #[test]
    fn truncate_works() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let mut file = check!(File::create(&path));
        check!(file.write(b"foo"));
        check!(file.sync_all());

        // Do some simple things with truncation
        assert_eq!(check!(file.metadata()).len(), 3);
        check!(file.set_len(10));
        assert_eq!(check!(file.metadata()).len(), 10);
        check!(file.write(b"bar"));
        check!(file.sync_all());
        assert_eq!(check!(file.metadata()).len(), 10);

        let mut v = Vec::new();
        check!(check!(File::open(&path)).read_to_end(&mut v));
        assert_eq!(v, b"foobar\0\0\0\0".to_vec());

        // Truncate to a smaller length, don't seek, and then write something.
        // Ensure that the intermediate zeroes are all filled in (we have `seek`ed
        // past the end of the file).
        check!(file.set_len(2));
        assert_eq!(check!(file.metadata()).len(), 2);
        check!(file.write(b"wut"));
        check!(file.sync_all());
        assert_eq!(check!(file.metadata()).len(), 9);
        let mut v = Vec::new();
        check!(check!(File::open(&path)).read_to_end(&mut v));
        assert_eq!(v, b"fo\0\0\0\0wut".to_vec());
    }

    #[test]
    fn open_flavors() {
        use fs::OpenOptions as OO;
        fn c<T: Clone>(t: &T) -> T { t.clone() }

        let tmpdir = tmpdir();

        let mut r = OO::new(); r.read(true);
        let mut w = OO::new(); w.write(true);
        let mut rw = OO::new(); rw.write(true).read(true);

        match r.open(&tmpdir.join("a")) {
            Ok(..) => panic!(), Err(..) => {}
        }

        // Perform each one twice to make sure that it succeeds the second time
        // (where the file exists)
        check!(c(&w).create(true).open(&tmpdir.join("b")));
        assert!(tmpdir.join("b").exists());
        check!(c(&w).create(true).open(&tmpdir.join("b")));
        check!(w.open(&tmpdir.join("b")));

        check!(c(&rw).create(true).open(&tmpdir.join("c")));
        assert!(tmpdir.join("c").exists());
        check!(c(&rw).create(true).open(&tmpdir.join("c")));
        check!(rw.open(&tmpdir.join("c")));

        check!(c(&w).append(true).create(true).open(&tmpdir.join("d")));
        assert!(tmpdir.join("d").exists());
        check!(c(&w).append(true).create(true).open(&tmpdir.join("d")));
        check!(c(&w).append(true).open(&tmpdir.join("d")));

        check!(c(&rw).append(true).create(true).open(&tmpdir.join("e")));
        assert!(tmpdir.join("e").exists());
        check!(c(&rw).append(true).create(true).open(&tmpdir.join("e")));
        check!(c(&rw).append(true).open(&tmpdir.join("e")));

        check!(c(&w).truncate(true).create(true).open(&tmpdir.join("f")));
        assert!(tmpdir.join("f").exists());
        check!(c(&w).truncate(true).create(true).open(&tmpdir.join("f")));
        check!(c(&w).truncate(true).open(&tmpdir.join("f")));

        check!(c(&rw).truncate(true).create(true).open(&tmpdir.join("g")));
        assert!(tmpdir.join("g").exists());
        check!(c(&rw).truncate(true).create(true).open(&tmpdir.join("g")));
        check!(c(&rw).truncate(true).open(&tmpdir.join("g")));

        check!(check!(File::create(&tmpdir.join("h"))).write("foo".as_bytes()));
        check!(r.open(&tmpdir.join("h")));
        {
            let mut f = check!(r.open(&tmpdir.join("h")));
            assert!(f.write("wut".as_bytes()).is_err());
        }
        assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 3);
        {
            let mut f = check!(c(&w).append(true).open(&tmpdir.join("h")));
            check!(f.write("bar".as_bytes()));
        }
        assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 6);
        {
            let mut f = check!(c(&w).truncate(true).open(&tmpdir.join("h")));
            check!(f.write("bar".as_bytes()));
        }
        assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 3);
    }

    #[test]
    fn binary_file() {
        let mut bytes = [0; 1024];
        StdRng::new().unwrap().fill_bytes(&mut bytes);

        let tmpdir = tmpdir();

        check!(check!(File::create(&tmpdir.join("test"))).write(&bytes));
        let mut v = Vec::new();
        check!(check!(File::open(&tmpdir.join("test"))).read_to_end(&mut v));
        assert!(v == &bytes[..]);
    }

    #[test]
    #[cfg(not(windows))]
    fn unlink_readonly() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("file");
        check!(File::create(&path));
        let mut perm = check!(fs::metadata(&path)).permissions();
        perm.set_readonly(true);
        check!(fs::set_permissions(&path, perm));
        check!(fs::remove_file(&path));
    }

    #[test]
    fn mkdir_trailing_slash() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("file");
        check!(fs::create_dir_all(&path.join("a/")));
    }

    #[test]
    fn canonicalize_works_simple() {
        let tmpdir = tmpdir();
        let tmpdir = fs::canonicalize(tmpdir.path()).unwrap();
        let file = tmpdir.join("test");
        File::create(&file).unwrap();
        assert_eq!(fs::canonicalize(&file).unwrap(), file);
    }

    #[test]
    #[cfg(not(windows))]
    fn realpath_works() {
        let tmpdir = tmpdir();
        let tmpdir = fs::canonicalize(tmpdir.path()).unwrap();
        let file = tmpdir.join("test");
        let dir = tmpdir.join("test2");
        let link = dir.join("link");
        let linkdir = tmpdir.join("test3");

        File::create(&file).unwrap();
        fs::create_dir(&dir).unwrap();
        fs::soft_link(&file, &link).unwrap();
        fs::soft_link(&dir, &linkdir).unwrap();

        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());

        assert_eq!(fs::canonicalize(&tmpdir).unwrap(), tmpdir);
        assert_eq!(fs::canonicalize(&file).unwrap(), file);
        assert_eq!(fs::canonicalize(&link).unwrap(), file);
        assert_eq!(fs::canonicalize(&linkdir).unwrap(), dir);
        assert_eq!(fs::canonicalize(&linkdir.join("link")).unwrap(), file);
    }

    #[test]
    #[cfg(not(windows))]
    fn realpath_works_tricky() {
        let tmpdir = tmpdir();
        let tmpdir = fs::canonicalize(tmpdir.path()).unwrap();

        let a = tmpdir.join("a");
        let b = a.join("b");
        let c = b.join("c");
        let d = a.join("d");
        let e = d.join("e");
        let f = a.join("f");

        fs::create_dir_all(&b).unwrap();
        fs::create_dir_all(&d).unwrap();
        File::create(&f).unwrap();
        fs::soft_link("../d/e", &c).unwrap();
        fs::soft_link("../f", &e).unwrap();

        assert_eq!(fs::canonicalize(&c).unwrap(), f);
        assert_eq!(fs::canonicalize(&e).unwrap(), f);
    }

    #[test]
    fn dir_entry_methods() {
        let tmpdir = tmpdir();

        fs::create_dir_all(&tmpdir.join("a")).unwrap();
        File::create(&tmpdir.join("b")).unwrap();

        for file in tmpdir.path().read_dir().unwrap().map(|f| f.unwrap()) {
            let fname = file.file_name();
            match fname.to_str() {
                Some("a") => {
                    assert!(file.file_type().unwrap().is_dir());
                    assert!(file.metadata().unwrap().is_dir());
                }
                Some("b") => {
                    assert!(file.file_type().unwrap().is_file());
                    assert!(file.metadata().unwrap().is_file());
                }
                f => panic!("unknown file name: {:?}", f),
            }
        }
    }

    #[test]
    fn read_dir_not_found() {
        let res = fs::read_dir("/path/that/does/not/exist");
        assert_eq!(res.err().unwrap().kind(), ErrorKind::NotFound);
    }
}
