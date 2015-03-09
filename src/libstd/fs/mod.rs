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

use core::prelude::*;

use io::{self, Error, ErrorKind, SeekFrom, Seek, Read, Write};
use path::{AsPath, Path, PathBuf};
use sys::fs2 as fs_imp;
use sys_common::{AsInnerMut, FromInner, AsInner};
use vec::Vec;

#[allow(deprecated)]
pub use self::tempdir::TempDir;

mod tempdir;

/// A reference to an open file on the filesystem.
///
/// An instance of a `File` can be read and/or written depending on what options
/// it was opened with. Files also implement `Seek` to alter the logical cursor
/// that the file contains internally.
///
/// # Example
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
    path: PathBuf,
}

/// Metadata information about a file.
///
/// This structure is returned from the `metadata` function or method and
/// represents known metadata about a file such as its permissions, size,
/// modification times, etc.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Metadata(fs_imp::FileAttr);

/// Iterator over the entries in a directory.
///
/// This iterator is returned from the `read_dir` function of this module and
/// will yield instances of `io::Result<DirEntry>`. Through a `DirEntry`
/// information like the entry's path and possibly other metadata can be
/// learned.
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
                     as symlinks differently")]
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

impl File {
    /// Attempts to open a file in read-only mode.
    ///
    /// See the `OpenOptions::open` method for more details.
    ///
    /// # Errors
    ///
    /// This function will return an error if `path` does not already exist.
    /// Other errors may also be returned according to `OpenOptions::open`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn open<P: AsPath + ?Sized>(path: &P) -> io::Result<File> {
        OpenOptions::new().read(true).open(path)
    }

    /// Open a file in write-only mode.
    ///
    /// This function will create a file if it does not exist,
    /// and will truncate it if it does.
    ///
    /// See the `OpenOptions::open` function for more details.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn create<P: AsPath + ?Sized>(path: &P) -> io::Result<File> {
        OpenOptions::new().write(true).create(true).truncate(true).open(path)
    }

    /// Returns the original path that was used to open this file.
    #[unstable(feature = "file_path",
               reason = "this abstraction is imposed by this library instead \
                         of the underlying OS and may be removed")]
    pub fn path(&self) -> Option<&Path> {
        Some(&self.path)
    }

    /// Attempt to sync all OS-internal metadata to disk.
    ///
    /// This function will attempt to ensure that all in-core data reaches the
    /// filesystem before returning.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set_len(&self, size: u64) -> io::Result<()> {
        self.inner.truncate(size)
    }

    /// Queries metadata about the underlying file.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn metadata(&self) -> io::Result<Metadata> {
        self.inner.file_attr().map(Metadata)
    }
}

impl AsInner<fs_imp::File> for File {
    fn as_inner(&self) -> &fs_imp::File { &self.inner }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Read for File {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> OpenOptions {
        OpenOptions(fs_imp::OpenOptions::new())
    }

    /// Set the option for read access.
    ///
    /// This option, when true, will indicate that the file should be
    /// `read`-able if opened.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn read(&mut self, read: bool) -> &mut OpenOptions {
        self.0.read(read); self
    }

    /// Set the option for write access.
    ///
    /// This option, when true, will indicate that the file should be
    /// `write`-able if opened.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write(&mut self, write: bool) -> &mut OpenOptions {
        self.0.write(write); self
    }

    /// Set the option for the append mode.
    ///
    /// This option, when true, means that writes will append to a file instead
    /// of overwriting previous contents.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn append(&mut self, append: bool) -> &mut OpenOptions {
        self.0.append(append); self
    }

    /// Set the option for truncating a previous file.
    ///
    /// If a file is successfully opened with this option set it will truncate
    /// the file to 0 length if it already exists.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, truncate: bool) -> &mut OpenOptions {
        self.0.truncate(truncate); self
    }

    /// Set the option for creating a new file.
    ///
    /// This option indicates whether a new file will be created if the file
    /// does not yet already exist.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn create(&mut self, create: bool) -> &mut OpenOptions {
        self.0.create(create); self
    }

    /// Open a file at `path` with the options specified by `self`.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn open<P: AsPath + ?Sized>(&self, path: &P) -> io::Result<File> {
        let path = path.as_path();
        let inner = try!(fs_imp::File::open(path, &self.0));
        Ok(File { path: path.to_path_buf(), inner: inner })
    }
}

impl AsInnerMut<fs_imp::OpenOptions> for OpenOptions {
    fn as_inner_mut(&mut self) -> &mut fs_imp::OpenOptions { &mut self.0 }
}

impl Metadata {
    /// Returns whether this metadata is for a directory.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_dir(&self) -> bool { self.0.is_dir() }

    /// Returns whether this metadata is for a regular file.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_file(&self) -> bool { self.0.is_file() }

    /// Returns the size of the file, in bytes, this metadata is for.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> u64 { self.0.size() }

    /// Returns the permissions of the file this metadata is for.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn permissions(&self) -> Permissions {
        Permissions(self.0.perm())
    }

    /// Returns the most recent access time for a file.
    ///
    /// The return value is in milliseconds since the epoch.
    #[unstable(feature = "fs_time",
               reason = "the return type of u64 is not quite appropriate for \
                         this method and may change if the standard library \
                         gains a type to represent a moment in time")]
    pub fn accessed(&self) -> u64 { self.0.accessed() }

    /// Returns the most recent modification time for a file.
    ///
    /// The return value is in milliseconds since the epoch.
    #[unstable(feature = "fs_time",
               reason = "the return type of u64 is not quite appropriate for \
                         this method and may change if the standard library \
                         gains a type to represent a moment in time")]
    pub fn modified(&self) -> u64 { self.0.modified() }
}

impl Permissions {
    /// Returns whether these permissions describe a readonly file.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn readonly(&self) -> bool { self.0.readonly() }

    /// Modify the readonly flag for this set of permissions.
    ///
    /// This operation does **not** modify the filesystem. To modify the
    /// filesystem use the `fs::set_permissions` function.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set_readonly(&mut self, readonly: bool) {
        self.0.set_readonly(readonly)
    }
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

#[stable(feature = "rust1", since = "1.0.0")]
impl DirEntry {
    /// Returns the full path to the file that this entry represents.
    ///
    /// The full path is created by joining the original path to `read_dir` or
    /// `walk_dir` with the filename of this entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn path(&self) -> PathBuf { self.0.path() }
}

/// Remove a file from the underlying filesystem.
///
/// # Example
///
/// ```rust,no_run
/// use std::fs;
///
/// fs::remove_file("/some/file/path.txt");
/// ```
///
/// Note that, just because an unlink call was successful, it is not
/// guaranteed that a file is immediately deleted (e.g. depending on
/// platform, other open file descriptors may prevent immediate removal).
///
/// # Errors
///
/// This function will return an error if `path` points to a directory, if the
/// user lacks permissions to remove the file, or if some other filesystem-level
/// error occurs.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn remove_file<P: AsPath + ?Sized>(path: &P) -> io::Result<()> {
    fs_imp::unlink(path.as_path())
}

/// Given a path, query the file system to get information about a file,
/// directory, etc.
///
/// This function will traverse soft links to query information about the
/// destination file.
///
/// # Example
///
/// ```rust,no_run
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
pub fn metadata<P: AsPath + ?Sized>(path: &P) -> io::Result<Metadata> {
    fs_imp::stat(path.as_path()).map(Metadata)
}

/// Rename a file or directory to a new name.
///
/// # Example
///
/// ```rust,no_run
/// use std::fs;
///
/// fs::rename("foo", "bar");
/// ```
///
/// # Errors
///
/// This function will return an error if the provided `from` doesn't exist, if
/// the process lacks permissions to view the contents, if `from` and `to`
/// reside on separate filesystems, or if some other intermittent I/O error
/// occurs.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn rename<P: AsPath + ?Sized, Q: AsPath + ?Sized>(from: &P, to: &Q)
                                                      -> io::Result<()> {
    fs_imp::rename(from.as_path(), to.as_path())
}

/// Copies the contents of one file to another. This function will also
/// copy the permission bits of the original file to the destination file.
///
/// This function will **overwrite** the contents of `to`.
///
/// Note that if `from` and `to` both point to the same file, then the file
/// will likely get truncated by this operation.
///
/// # Example
///
/// ```rust
/// use std::fs;
///
/// fs::copy("foo.txt", "bar.txt");
/// ```
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<P: AsPath + ?Sized, Q: AsPath + ?Sized>(from: &P, to: &Q)
                                                    -> io::Result<u64> {
    let from = from.as_path();
    if !from.is_file() {
        return Err(Error::new(ErrorKind::MismatchedFileTypeForOperation,
                              "the source path is not an existing file",
                              None))
    }

    let mut reader = try!(File::open(from));
    let mut writer = try!(File::create(to));
    let perm = try!(reader.metadata()).permissions();

    let ret = try!(io::copy(&mut reader, &mut writer));
    try!(set_permissions(to, perm));
    Ok(ret)
}

/// Creates a new hard link on the filesystem.
///
/// The `dst` path will be a link pointing to the `src` path. Note that systems
/// often require these two paths to both be located on the same filesystem.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn hard_link<P: AsPath + ?Sized, Q: AsPath + ?Sized>(src: &P, dst: &Q)
                                                         -> io::Result<()> {
    fs_imp::link(src.as_path(), dst.as_path())
}

/// Creates a new soft link on the filesystem.
///
/// The `dst` path will be a soft link pointing to the `src` path.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn soft_link<P: AsPath + ?Sized, Q: AsPath + ?Sized>(src: &P, dst: &Q)
                                                         -> io::Result<()> {
    fs_imp::symlink(src.as_path(), dst.as_path())
}

/// Reads a soft link, returning the file that the link points to.
///
/// # Errors
///
/// This function will return an error on failure. Failure conditions include
/// reading a file that does not exist or reading a file that is not a soft
/// link.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn read_link<P: AsPath + ?Sized>(path: &P) -> io::Result<PathBuf> {
    fs_imp::readlink(path.as_path())
}

/// Create a new, empty directory at the provided path
///
/// # Example
///
/// ```rust
/// use std::fs;
///
/// fs::create_dir("/some/dir");
/// ```
///
/// # Errors
///
/// This function will return an error if the user lacks permissions to make a
/// new directory at the provided `path`, or if the directory already exists.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn create_dir<P: AsPath + ?Sized>(path: &P) -> io::Result<()> {
    fs_imp::mkdir(path.as_path())
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn create_dir_all<P: AsPath + ?Sized>(path: &P) -> io::Result<()> {
    let path = path.as_path();
    if path.is_dir() { return Ok(()) }
    if let Some(p) = path.parent() { try!(create_dir_all(p)) }
    create_dir(path)
}

/// Remove an existing, empty directory
///
/// # Example
///
/// ```rust
/// use std::fs;
///
/// fs::remove_dir("/some/dir");
/// ```
///
/// # Errors
///
/// This function will return an error if the user lacks permissions to remove
/// the directory at the provided `path`, or if the directory isn't empty.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn remove_dir<P: AsPath + ?Sized>(path: &P) -> io::Result<()> {
    fs_imp::rmdir(path.as_path())
}

/// Removes a directory at this path, after removing all its contents. Use
/// carefully!
///
/// This function does **not** follow soft links and it will simply remove the
/// soft link itself.
///
/// # Errors
///
/// See `file::remove_file` and `fs::remove_dir`
#[stable(feature = "rust1", since = "1.0.0")]
pub fn remove_dir_all<P: AsPath + ?Sized>(path: &P) -> io::Result<()> {
    let path = path.as_path();
    for child in try!(read_dir(path)) {
        let child = try!(child).path();
        let stat = try!(lstat(&*child));
        if stat.is_dir() {
            try!(remove_dir_all(&*child));
        } else {
            try!(remove_file(&*child));
        }
    }
    return remove_dir(path);

    #[cfg(unix)]
    fn lstat(path: &Path) -> io::Result<fs_imp::FileAttr> { fs_imp::lstat(path) }
    #[cfg(windows)]
    fn lstat(path: &Path) -> io::Result<fs_imp::FileAttr> { fs_imp::stat(path) }
}

/// Returns an iterator over the entries within a directory.
///
/// The iterator will yield instances of `io::Result<DirEntry>`. New errors may
/// be encountered after an iterator is initially constructed.
///
/// # Example
///
/// ```rust
/// use std::io;
/// use std::fs::{self, PathExt, DirEntry};
/// use std::path::Path;
///
/// // one possible implementation of fs::walk_dir only visiting files
/// fn visit_dirs(dir: &Path, cb: &mut FnMut(DirEntry)) -> io::Result<()> {
///     if dir.is_dir() {
///         for entry in try!(fs::read_dir(dir)) {
///             let entry = try!(entry);
///             if entry.path().is_dir() {
///                 try!(visit_dirs(&entry.path(), cb));
///             } else {
///                 cb(entry);
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
pub fn read_dir<P: AsPath + ?Sized>(path: &P) -> io::Result<ReadDir> {
    fs_imp::readdir(path.as_path()).map(ReadDir)
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
                     as symlinks differently")]
pub fn walk_dir<P: AsPath + ?Sized>(path: &P) -> io::Result<WalkDir> {
    let start = try!(read_dir(path));
    Ok(WalkDir { cur: Some(start), stack: Vec::new() })
}

#[unstable(feature = "fs_walk")]
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
#[unstable(feature = "path_ext",
           reason = "the precise set of methods exposed on this trait may \
                     change and some methods may be removed")]
pub trait PathExt {
    /// Get information on the file, directory, etc at this path.
    ///
    /// Consult the `fs::stat` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with `file::stat`.
    fn metadata(&self) -> io::Result<Metadata>;

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

impl PathExt for Path {
    fn metadata(&self) -> io::Result<Metadata> { metadata(self) }

    fn exists(&self) -> bool { metadata(self).is_ok() }

    fn is_file(&self) -> bool {
        metadata(self).map(|s| s.is_file()).unwrap_or(false)
    }
    fn is_dir(&self) -> bool {
        metadata(self).map(|s| s.is_dir()).unwrap_or(false)
    }
}

/// Changes the timestamps for a file's last modification and access time.
///
/// The file at the path specified will have its last access time set to
/// `atime` and its modification time set to `mtime`. The times specified should
/// be in milliseconds.
#[unstable(feature = "fs_time",
           reason = "the argument type of u64 is not quite appropriate for \
                     this function and may change if the standard library \
                     gains a type to represent a moment in time")]
pub fn set_file_times<P: AsPath + ?Sized>(path: &P, accessed: u64,
                                          modified: u64) -> io::Result<()> {
    fs_imp::utimes(path.as_path(), accessed, modified)
}

/// Changes the permissions found on a file or a directory.
///
/// # Example
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
#[unstable(feature = "fs",
           reason = "a more granual ability to set specific permissions may \
                     be exposed on the Permissions structure itself and this \
                     method may not always exist")]
pub fn set_permissions<P: AsPath + ?Sized>(path: &P, perm: Permissions)
                                           -> io::Result<()> {
    fs_imp::set_perm(path.as_path(), perm.0)
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)] //rand

    use prelude::v1::*;
    use io::prelude::*;

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
        let s = os::tmpdir();
        let p = Path2::new(s.as_str().unwrap());
        let ret = p.join(&format!("rust-{}", rand::random::<u32>()));
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
                -1|0 => panic!("shouldn't happen"),
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
        for n in range(0, 3) {
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
        assert_eq!(e.kind(), ErrorKind::PathAlreadyExists);
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
        dirpath.push(&format!("test-가一ー你好"));
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
    fn copy_file_ok() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        check!(check!(File::create(&input)).write(b"hello"));
        check!(fs::copy(&input, &out));
        let mut v = Vec::new();
        check!(check!(File::open(&out)).read_to_end(&mut v));
        assert_eq!(v.as_slice(), b"hello");

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
        // Ensure that the intermediate zeroes are all filled in (we're seeked
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
    fn utime() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("a");
        check!(File::create(&path));
        // These numbers have to be bigger than the time in the day to account
        // for timezones Windows in particular will fail in certain timezones
        // with small enough values
        check!(fs::set_file_times(&path, 100000, 200000));
        assert_eq!(check!(path.metadata()).accessed(), 100000);
        assert_eq!(check!(path.metadata()).modified(), 200000);
    }

    #[test]
    fn utime_noexist() {
        let tmpdir = tmpdir();

        match fs::set_file_times(&tmpdir.join("a"), 100, 200) {
            Ok(..) => panic!(),
            Err(..) => {}
        }
    }

    #[test]
    fn binary_file() {
        let mut bytes = [0; 1024];
        StdRng::new().ok().unwrap().fill_bytes(&mut bytes);

        let tmpdir = tmpdir();

        check!(check!(File::create(&tmpdir.join("test"))).write(&bytes));
        let mut v = Vec::new();
        check!(check!(File::open(&tmpdir.join("test"))).read_to_end(&mut v));
        assert!(v == bytes.as_slice());
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
}
