// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

/*! Synchronous File I/O

This module provides a set of functions and traits for working
with regular files & directories on a filesystem.

At the top-level of the module are a set of freestanding functions, associated
with various filesystem operations. They all operate on `Path` objects.

All operations in this module, including those as part of `File` et al
block the task during execution. In the event of failure, all functions/methods
will return an `IoResult` type with an `Err` value.

Also included in this module is an implementation block on the `Path` object
defined in `std::path::Path`. The impl adds useful methods about inspecting the
metadata of a file. This includes getting the `stat` information, reading off
particular bits of it, etc.

# Example

```rust
# #![allow(unused_must_use)]
use std::io::fs::PathExtensions;
use std::io::{File, fs};

let path = Path::new("foo.txt");

// create the file, whether it exists or not
let mut file = File::create(&path);
file.write(b"foobar");
# drop(file);

// open the file in read-only mode
let mut file = File::open(&path);
file.read_to_end();

println!("{}", path.stat().unwrap().size);
# drop(file);
fs::unlink(&path);
```

*/

use clone::Clone;
use io::standard_error;
use io::{FilePermission, Write, Open, FileAccess, FileMode};
use io::{IoResult, IoError, FileStat, SeekStyle, Seek, Writer, Reader};
use io::{Read, Truncate, ReadWrite, Append};
use io::UpdateIoError;
use io;
use iter::{Iterator, Extend};
use option::{Some, None, Option};
use path::{Path, GenericPath};
use path;
use result::{Err, Ok};
use slice::SlicePrelude;
use string::String;
use vec::Vec;

use sys::fs as fs_imp;
use sys_common;

/// Unconstrained file access type that exposes read and write operations
///
/// Can be constructed via `File::open()`, `File::create()`, and
/// `File::open_mode()`.
///
/// # Error
///
/// This type will return errors as an `IoResult<T>` if operations are
/// attempted against it for which its underlying file descriptor was not
/// configured at creation time, via the `FileAccess` parameter to
/// `File::open_mode()`.
pub struct File {
    fd: fs_imp::FileDesc,
    path: Path,
    last_nread: int,
}

impl sys_common::AsFileDesc for File {
    fn as_fd(&self) -> &fs_imp::FileDesc { unimplemented!() }
}

impl File {
    /// Open a file at `path` in the mode specified by the `mode` and `access`
    /// arguments
    ///
    /// # Example
    ///
    /// ```rust,should_fail
    /// use std::io::{File, Open, ReadWrite};
    ///
    /// let p = Path::new("/some/file/path.txt");
    ///
    /// let file = match File::open_mode(&p, Open, ReadWrite) {
    ///     Ok(f) => f,
    ///     Err(e) => panic!("file error: {}", e),
    /// };
    /// // do some stuff with that file
    ///
    /// // the file will be closed at the end of this block
    /// ```
    ///
    /// `FileMode` and `FileAccess` provide information about the permissions
    /// context in which a given stream is created. More information about them
    /// can be found in `std::io`'s docs. If a file is opened with `Write`
    /// or `ReadWrite` access, then it will be created if it does not already
    /// exist.
    ///
    /// Note that, with this function, a `File` is returned regardless of the
    /// access-limitations indicated by `FileAccess` (e.g. calling `write` on a
    /// `File` opened as `Read` will return an error at runtime).
    ///
    /// # Error
    ///
    /// This function will return an error under a number of different
    /// circumstances, to include but not limited to:
    ///
    /// * Opening a file that does not exist with `Read` access.
    /// * Attempting to open a file with a `FileAccess` that the user lacks
    ///   permissions for
    /// * Filesystem-level errors (full disk, etc)
    pub fn open_mode(path: &Path,
                     mode: FileMode,
                     access: FileAccess) -> IoResult<File> { unimplemented!() }

    /// Attempts to open a file in read-only mode. This function is equivalent to
    /// `File::open_mode(path, Open, Read)`, and will raise all of the same
    /// errors that `File::open_mode` does.
    ///
    /// For more information, see the `File::open_mode` function.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::io::File;
    ///
    /// let contents = File::open(&Path::new("foo.txt")).read_to_end();
    /// ```
    pub fn open(path: &Path) -> IoResult<File> { unimplemented!() }

    /// Attempts to create a file in write-only mode. This function is
    /// equivalent to `File::open_mode(path, Truncate, Write)`, and will
    /// raise all of the same errors that `File::open_mode` does.
    ///
    /// For more information, see the `File::open_mode` function.
    ///
    /// # Example
    ///
    /// ```rust
    /// # #![allow(unused_must_use)]
    /// use std::io::File;
    ///
    /// let mut f = File::create(&Path::new("foo.txt"));
    /// f.write(b"This is a sample file");
    /// # drop(f);
    /// # ::std::io::fs::unlink(&Path::new("foo.txt"));
    /// ```
    pub fn create(path: &Path) -> IoResult<File> { unimplemented!() }

    /// Returns the original path which was used to open this file.
    pub fn path<'a>(&'a self) -> &'a Path { unimplemented!() }

    /// Synchronizes all modifications to this file to its permanent storage
    /// device. This will flush any internal buffers necessary to perform this
    /// operation.
    pub fn fsync(&mut self) -> IoResult<()> { unimplemented!() }

    /// This function is similar to `fsync`, except that it may not synchronize
    /// file metadata to the filesystem. This is intended for use case which
    /// must synchronize content, but don't need the metadata on disk. The goal
    /// of this method is to reduce disk operations.
    pub fn datasync(&mut self) -> IoResult<()> { unimplemented!() }

    /// Either truncates or extends the underlying file, updating the size of
    /// this file to become `size`. This is equivalent to unix's `truncate`
    /// function.
    ///
    /// If the `size` is less than the current file's size, then the file will
    /// be shrunk. If it is greater than the current file's size, then the file
    /// will be extended to `size` and have all of the intermediate data filled
    /// in with 0s.
    pub fn truncate(&mut self, size: i64) -> IoResult<()> { unimplemented!() }

    /// Returns true if the stream has reached the end of the file.
    ///
    /// If true, then this file will no longer continue to return data via
    /// `read`.
    ///
    /// Note that the operating system will not return an `EOF` indicator
    /// until you have attempted to read past the end of the file, so if
    /// you've read _exactly_ the number of bytes in the file, this will
    /// return `false`, not `true`.
    pub fn eof(&self) -> bool { unimplemented!() }

    /// Queries information about the underlying file.
    pub fn stat(&mut self) -> IoResult<FileStat> { unimplemented!() }
}

/// Unlink a file from the underlying filesystem.
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::fs;
///
/// let p = Path::new("/some/file/path.txt");
/// fs::unlink(&p);
/// ```
///
/// Note that, just because an unlink call was successful, it is not
/// guaranteed that a file is immediately deleted (e.g. depending on
/// platform, other open file descriptors may prevent immediate removal)
///
/// # Error
///
/// This function will return an error if `path` points to a directory, if the
/// user lacks permissions to remove the file, or if some other filesystem-level
/// error occurs.
pub fn unlink(path: &Path) -> IoResult<()> { unimplemented!() }

/// Given a path, query the file system to get information about a file,
/// directory, etc. This function will traverse symlinks to query
/// information about the destination file.
///
/// # Example
///
/// ```rust
/// use std::io::fs;
///
/// let p = Path::new("/some/file/path.txt");
/// match fs::stat(&p) {
///     Ok(stat) => { /* ... */ }
///     Err(e) => { /* handle error */ }
/// }
/// ```
///
/// # Error
///
/// This function will return an error if the user lacks the requisite permissions
/// to perform a `stat` call on the given `path` or if there is no entry in the
/// filesystem at the provided path.
pub fn stat(path: &Path) -> IoResult<FileStat> { unimplemented!() }

/// Perform the same operation as the `stat` function, except that this
/// function does not traverse through symlinks. This will return
/// information about the symlink file instead of the file that it points
/// to.
///
/// # Error
///
/// See `stat`
pub fn lstat(path: &Path) -> IoResult<FileStat> { unimplemented!() }

/// Rename a file or directory to a new name.
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::fs;
///
/// fs::rename(&Path::new("foo"), &Path::new("bar"));
/// ```
///
/// # Error
///
/// This function will return an error if the provided `from` doesn't exist, if
/// the process lacks permissions to view the contents, or if some other
/// intermittent I/O error occurs.
pub fn rename(from: &Path, to: &Path) -> IoResult<()> { unimplemented!() }

/// Copies the contents of one file to another. This function will also
/// copy the permission bits of the original file to the destination file.
///
/// Note that if `from` and `to` both point to the same file, then the file
/// will likely get truncated by this operation.
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::fs;
///
/// fs::copy(&Path::new("foo.txt"), &Path::new("bar.txt"));
/// ```
///
/// # Error
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The `from` path is not a file
/// * The `from` file does not exist
/// * The current process does not have the permission rights to access
///   `from` or write `to`
///
/// Note that this copy is not atomic in that once the destination is
/// ensured to not exist, there is nothing preventing the destination from
/// being created and then destroyed by this operation.
pub fn copy(from: &Path, to: &Path) -> IoResult<()> { unimplemented!() }

/// Changes the permission mode bits found on a file or a directory. This
/// function takes a mask from the `io` module
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io;
/// use std::io::fs;
///
/// fs::chmod(&Path::new("file.txt"), io::USER_FILE);
/// fs::chmod(&Path::new("file.txt"), io::USER_READ | io::USER_WRITE);
/// fs::chmod(&Path::new("dir"),      io::USER_DIR);
/// fs::chmod(&Path::new("file.exe"), io::USER_EXEC);
/// ```
///
/// # Error
///
/// This function will return an error if the provided `path` doesn't exist, if
/// the process lacks permissions to change the attributes of the file, or if
/// some other I/O error is encountered.
pub fn chmod(path: &Path, mode: io::FilePermission) -> IoResult<()> { unimplemented!() }

/// Change the user and group owners of a file at the specified path.
pub fn chown(path: &Path, uid: int, gid: int) -> IoResult<()> {
    fs_imp::chown(path, uid, gid)
           .update_err("couldn't chown path", |e|
               format!("{}; path={}; uid={}; gid={}", e, path.display(), uid, gid))
}

/// Creates a new hard link on the filesystem. The `dst` path will be a
/// link pointing to the `src` path. Note that systems often require these
/// two paths to both be located on the same filesystem.
pub fn link(src: &Path, dst: &Path) -> IoResult<()> { unimplemented!() }

/// Creates a new symbolic link on the filesystem. The `dst` path will be a
/// symlink pointing to the `src` path.
pub fn symlink(src: &Path, dst: &Path) -> IoResult<()> { unimplemented!() }

/// Reads a symlink, returning the file that the symlink points to.
///
/// # Error
///
/// This function will return an error on failure. Failure conditions include
/// reading a file that does not exist or reading a file which is not a symlink.
pub fn readlink(path: &Path) -> IoResult<Path> { unimplemented!() }

/// Create a new, empty directory at the provided path
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io;
/// use std::io::fs;
///
/// let p = Path::new("/some/dir");
/// fs::mkdir(&p, io::USER_RWX);
/// ```
///
/// # Error
///
/// This function will return an error if the user lacks permissions to make a
/// new directory at the provided `path`, or if the directory already exists.
pub fn mkdir(path: &Path, mode: FilePermission) -> IoResult<()> { unimplemented!() }

/// Remove an existing, empty directory
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::fs;
///
/// let p = Path::new("/some/dir");
/// fs::rmdir(&p);
/// ```
///
/// # Error
///
/// This function will return an error if the user lacks permissions to remove
/// the directory at the provided `path`, or if the directory isn't empty.
pub fn rmdir(path: &Path) -> IoResult<()> { unimplemented!() }

/// Retrieve a vector containing all entries within a provided directory
///
/// # Example
///
/// ```rust
/// use std::io::fs::PathExtensions;
/// use std::io::fs;
/// use std::io;
///
/// // one possible implementation of fs::walk_dir only visiting files
/// fn visit_dirs(dir: &Path, cb: |&Path|) -> io::IoResult<()> {
///     if dir.is_dir() {
///         let contents = try!(fs::readdir(dir));
///         for entry in contents.iter() {
///             if entry.is_dir() {
///                 try!(visit_dirs(entry, |p| cb(p)));
///             } else {
///                 cb(entry);
///             }
///         }
///         Ok(())
///     } else {
///         Err(io::standard_error(io::InvalidInput))
///     }
/// }
/// ```
///
/// # Error
///
/// This function will return an error if the provided `path` doesn't exist, if
/// the process lacks permissions to view the contents or if the `path` points
/// at a non-directory file
pub fn readdir(path: &Path) -> IoResult<Vec<Path>> { unimplemented!() }

/// Returns an iterator which will recursively walk the directory structure
/// rooted at `path`. The path given will not be iterated over, and this will
/// perform iteration in some top-down order.  The contents of unreadable
/// subdirectories are ignored.
pub fn walk_dir(path: &Path) -> IoResult<Directories> { unimplemented!() }

/// An iterator which walks over a directory
pub struct Directories {
    stack: Vec<Path>,
}

impl Iterator<Path> for Directories {
    fn next(&mut self) -> Option<Path> { unimplemented!() }
}

/// Recursively create a directory and all of its parent components if they
/// are missing.
///
/// # Error
///
/// See `fs::mkdir`.
pub fn mkdir_recursive(path: &Path, mode: FilePermission) -> IoResult<()> { unimplemented!() }

/// Removes a directory at this path, after removing all its contents. Use
/// carefully!
///
/// # Error
///
/// See `file::unlink` and `fs::readdir`
pub fn rmdir_recursive(path: &Path) -> IoResult<()> { unimplemented!() }

/// Changes the timestamps for a file's last modification and access time.
/// The file at the path specified will have its last access time set to
/// `atime` and its modification time set to `mtime`. The times specified should
/// be in milliseconds.
// FIXME(#10301) these arguments should not be u64
pub fn change_file_times(path: &Path, atime: u64, mtime: u64) -> IoResult<()> { unimplemented!() }

impl Reader for File {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl Writer for File {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
}

impl Seek for File {
    fn tell(&self) -> IoResult<u64> { unimplemented!() }

    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> { unimplemented!() }
}

/// Utility methods for paths.
pub trait PathExtensions {
    /// Get information on the file, directory, etc at this path.
    ///
    /// Consult the `fs::stat` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with `file::stat`.
    fn stat(&self) -> IoResult<FileStat>;

    /// Get information on the file, directory, etc at this path, not following
    /// symlinks.
    ///
    /// Consult the `fs::lstat` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with `file::lstat`.
    fn lstat(&self) -> IoResult<FileStat>;

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

impl PathExtensions for path::Path {
    fn stat(&self) -> IoResult<FileStat> { unimplemented!() }
    fn lstat(&self) -> IoResult<FileStat> { unimplemented!() }
    fn exists(&self) -> bool { unimplemented!() }
    fn is_file(&self) -> bool { unimplemented!() }
    fn is_dir(&self) -> bool { unimplemented!() }
}

fn mode_string(mode: FileMode) -> &'static str { unimplemented!() }

fn access_string(access: FileAccess) -> &'static str { unimplemented!() }
