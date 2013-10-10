// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Synchronous File I/O

This module provides a set of functions and traits for working
with regular files & directories on a filesystem.

At the top-level of the module are a set of freestanding functions,
associated with various filesystem operations. They all operate
on a `PathLike` object.

All operations in this module, including those as part of `FileStream` et al
block the task during execution. Most will raise `std::rt::io::{io_error,read_error}`
conditions in the event of failure.

Also included in this module are the `FileInfo` and `DirectoryInfo` traits. When
`use`'d alongside a value whose type implements them (A `std::path::Path` impl is
a part of this module), they expose a set of functions for operations against
a given file location, depending on whether the path already exists. Whenever
possible, the `{FileInfo, DirectoryInfo}` preserve the same semantics as their
free function counterparts.
*/

use prelude::*;
use super::support::PathLike;
use super::{Reader, Writer, Seek};
use super::{SeekStyle, Read, Write};
use rt::rtio::{RtioFileStream, IoFactory, IoFactoryObject};
use rt::io::{io_error, read_error, EndOfFile,
            FileMode, FileAccess, FileStat, IoError,
            PathAlreadyExists, PathDoesntExist,
            MismatchedFileTypeForOperation, ignore_io_error};
use rt::local::Local;
use option::{Some, None};
use path::Path;

/// Open a file for reading/writing, as indicated by `path`.
///
/// # Example
///
///     use std;
///     use std::path::Path;
///     use std::rt::io::support::PathLike;
///     use std::rt::io::file::open;
///     use std::rt::io::{FileMode, FileAccess};
///
///     let p = &Path("/some/file/path.txt");
///
///     do io_error::cond.trap(|_| {
///         // hoo-boy...
///     }).inside {
///         let stream = match open(p, Create, ReadWrite) {
///             Some(s) => s,
///             None => fail2!("whoops! I'm sure this raised, anyways..");
///         }
///         // do some stuff with that stream
///
///         // the file stream will be closed at the end of this block
///     }
///     // ..
///
/// `FileMode` and `FileAccess` provide information about the permissions
/// context in which a given stream is created. More information about them
/// can be found in `std::rt::io`'s docs.
///
/// Note that, with this function, a `FileStream` is returned regardless of
/// the access-limitations indicated by `FileAccess` (e.g. calling `write` on a
/// `FileStream` opened as `ReadOnly` will raise an `io_error` condition at runtime). If you
/// desire a more-correctly-constrained interface to files, use the
/// `{open_stream, open_reader, open_writer}` methods that are a part of `FileInfo`
///
/// # Errors
///
/// This function will raise an `io_error` condition under a number of different circumstances,
/// to include but not limited to:
///
/// * Opening a file that already exists with `FileMode` of `Create` or vice versa (e.g.
///   opening a non-existant file with `FileMode` or `Open`)
/// * Attempting to open a file with a `FileAccess` that the user lacks permissions
///   for
/// * Filesystem-level errors (full disk, etc)
pub fn open<P: PathLike>(path: &P,
                         mode: FileMode,
                         access: FileAccess
                        ) -> Option<FileStream> {
    let open_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_open(path, mode, access)
    };
    match open_result {
        Ok(fd) => Some(FileStream {
            fd: fd,
            last_nread: -1
        }),
        Err(ioerr) => {
            io_error::cond.raise(ioerr);
            None
        }
    }
}

/// Unlink a file from the underlying filesystem.
///
/// # Example
///
///     use std;
///     use std::path::Path;
///     use std::rt::io::support::PathLike;
///     use std::rt::io::file::unlink;
///
///     let p = &Path("/some/file/path.txt");
///     unlink(p);
///     // if we made it here without failing, then the
///     // unlink operation was successful
///
/// Note that, just because an unlink call was successful, it is not
/// guaranteed that a file is immediately deleted (e.g. depending on
/// platform, other open file descriptors may prevent immediate removal)
///
/// # Errors
///
/// This function will raise an `io_error` condition if the user lacks permissions to
/// remove the file or if some other filesystem-level error occurs
pub fn unlink<P: PathLike>(path: &P) {
    let unlink_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_unlink(path)
    };
    match unlink_result {
        Ok(_) => (),
        Err(ioerr) => {
            io_error::cond.raise(ioerr);
        }
    }
}

/// Create a new, empty directory at the provided path
///
/// # Example
///
///     use std;
///     use std::path::Path;
///     use std::rt::io::support::PathLike;
///     use std::rt::io::file::mkdir;
///
///     let p = &Path("/some/dir");
///     mkdir(p);
///     // If we got here, our directory exists! Horray!
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks permissions to make a
/// new directory at the provided path, or if the directory already exists
pub fn mkdir<P: PathLike>(path: &P) {
    let mkdir_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_mkdir(path)
    };
    match mkdir_result {
        Ok(_) => (),
        Err(ioerr) => {
            io_error::cond.raise(ioerr);
        }
    }
}

/// Remove an existing, empty directory
///
/// # Example
///
///     use std;
///     use std::path::Path;
///     use std::rt::io::support::PathLike;
///     use std::rt::io::file::rmdir;
///
///     let p = &Path("/some/dir");
///     rmdir(p);
///     // good riddance, you mean ol' directory
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks permissions to remove the
/// directory at the provided path, or if the directory isn't empty
pub fn rmdir<P: PathLike>(path: &P) {
    let rmdir_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_rmdir(path)
    };
    match rmdir_result {
        Ok(_) => (),
        Err(ioerr) => {
            io_error::cond.raise(ioerr);
        }
    }
}

/// Get information on the file, directory, etc at the provided path
///
/// Given a `rt::io::support::PathLike`, query the file system to get
/// information about a file, directory, etc.
///
/// Returns a `Some(std::rt::io::PathInfo)` on success
///
/// # Example
///
///     use std;
///     use std::path::Path;
///     use std::rt::io::support::PathLike;
///     use std::rt::io::file::stat;
///
///     let p = &Path("/some/file/path.txt");
///
///     do io_error::cond.trap(|_| {
///         // hoo-boy...
///     }).inside {
///         let info = match stat(p) {
///             Some(s) => s,
///             None => fail2!("whoops! I'm sure this raised, anyways..");
///         }
///         if stat.is_file {
///             // just imagine the possibilities ...
///         }
///
///         // the file stream will be closed at the end of this block
///     }
///     // ..
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks the requisite
/// permissions to perform a `stat` call on the given path or if there is no
/// entry in the filesystem at the provided path.
pub fn stat<P: PathLike>(path: &P) -> Option<FileStat> {
    let open_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_stat(path)
    };
    match open_result {
        Ok(p) => {
            Some(p)
        },
        Err(ioerr) => {
            io_error::cond.raise(ioerr);
            None
        }
    }
}

/// Retrieve a vector containing all entries within a provided directory
///
/// # Example
///
///     use std;
///     use std::path::Path;
///     use std::rt::io::support::PathLike;
///     use std::rt::io::file::readdir;
///
///     fn visit_dirs(dir: &Path, cb: &fn(&Path)) {
///         if dir.is_dir() {
///             let contents = dir.readdir();
///             for entry in contents.iter() {
///                 if entry.is_dir() { visit_dirs(entry, cb); }
///                 else { cb(entry); }
///             }
///         }
///         else { fail2!("nope"); }
///     }
///
/// # Errors
///
/// Will raise an `io_error` condition if the provided `path` doesn't exist,
/// the process lacks permissions to view the contents or if the `path` points
/// at a non-directory file
pub fn readdir<P: PathLike>(path: &P) -> Option<~[Path]> {
    let readdir_result = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_readdir(path, 0)
    };
    match readdir_result {
        Ok(p) => {
            Some(p)
        },
        Err(ioerr) => {
            io_error::cond.raise(ioerr);
            None
        }
    }
}

/// Constrained version of `FileStream` that only exposes read-specific operations.
///
/// Can be retreived via `FileInfo.open_reader()`.
pub struct FileReader { priv stream: FileStream }

/// a `std::rt::io::Reader` trait impl for file I/O.
impl Reader for FileReader {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        self.stream.read(buf)
    }

    fn eof(&mut self) -> bool {
        self.stream.eof()
    }
}

/// a `std::rt::io::Seek` trait impl for file I/O.
impl Seek for FileReader {
    fn tell(&self) -> u64 {
        self.stream.tell()
    }

    fn seek(&mut self, pos: i64, style: SeekStyle) {
        self.stream.seek(pos, style);
    }
}

/// Constrained version of `FileStream` that only exposes write-specific operations.
///
/// Can be retreived via `FileInfo.open_writer()`.
pub struct FileWriter { priv stream: FileStream }

/// a `std::rt::io::Writer` trait impl for file I/O.
impl Writer for FileWriter {
    fn write(&mut self, buf: &[u8]) {
        self.stream.write(buf);
    }

    fn flush(&mut self) {
        self.stream.flush();
    }
}

/// a `std::rt::io::Seek` trait impl for file I/O.
impl Seek for FileWriter {
    fn tell(&self) -> u64 {
        self.stream.tell()
    }

    fn seek(&mut self, pos: i64, style: SeekStyle) {
        self.stream.seek(pos, style);
    }
}

/// Unconstrained file access type that exposes read and write operations
///
/// Can be retreived via `file::open()` and `FileInfo.open_stream()`.
///
/// # Errors
///
/// This type will raise an io_error condition if operations are attempted against
/// it for which its underlying file descriptor was not configured at creation
/// time, via the `FileAccess` parameter to `file::open()`.
///
/// For this reason, it is best to use the access-constrained wrappers that are
/// exposed via `FileInfo.open_reader()` and `FileInfo.open_writer()`.
pub struct FileStream {
    fd: ~RtioFileStream,
    last_nread: int,
}

/// a `std::rt::io::Reader` trait impl for file I/O.
impl Reader for FileStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match self.fd.read(buf) {
            Ok(read) => {
                self.last_nread = read;
                match read {
                    0 => None,
                    _ => Some(read as uint)
                }
            },
            Err(ioerr) => {
                // EOF is indicated by returning None
                if ioerr.kind != EndOfFile {
                    read_error::cond.raise(ioerr);
                }
                return None;
            }
        }
    }

    fn eof(&mut self) -> bool {
        self.last_nread == 0
    }
}

/// a `std::rt::io::Writer` trait impl for file I/O.
impl Writer for FileStream {
    fn write(&mut self, buf: &[u8]) {
        match self.fd.write(buf) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }

    fn flush(&mut self) {
        match self.fd.flush() {
            Ok(_) => (),
            Err(ioerr) => {
                read_error::cond.raise(ioerr);
            }
        }
    }
}

/// a `std::rt::io:Seek` trait impl for file I/O.
impl Seek for FileStream {
    fn tell(&self) -> u64 {
        let res = self.fd.tell();
        match res {
            Ok(cursor) => cursor,
            Err(ioerr) => {
                read_error::cond.raise(ioerr);
                return -1;
            }
        }
    }

    fn seek(&mut self, pos: i64, style: SeekStyle) {
        match self.fd.seek(pos, style) {
            Ok(_) => {
                // successful seek resets EOF indicator
                self.last_nread = -1;
                ()
            },
            Err(ioerr) => {
                read_error::cond.raise(ioerr);
            }
        }
    }
}

/// Shared functionality between `FileInfo` and `DirectoryInfo`
pub trait FileSystemInfo {
    /// Get the filesystem path that this instance points at,
    /// whether it is valid or not. In this way, it can be used to
    /// to specify a path of a non-existent file which it
    /// later creates
    fn get_path<'a>(&'a self) -> &'a Path;

    /// Get information on the file, directory, etc at the provided path
    ///
    /// Consult the `file::stat` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with `file::stat`
    fn stat(&self) -> Option<FileStat> {
        stat(self.get_path())
    }

    /// Boolean value indicator whether the underlying file exists on the filesystem
    ///
    /// # Errors
    ///
    /// Will not raise a condition
    fn exists(&self) -> bool {
        match ignore_io_error(|| self.stat()) {
            Some(_) => true,
            None => false
        }
    }

}

/// Represents a file, whose underlying path may or may not be valid
///
/// # Example
///
/// * Check if a file exists, reading from it if so
///
/// ```rust
/// use std;
/// use std::path::Path;
/// use std::rt::io::file::{FileInfo, FileReader};
///
/// let f = &Path("/some/file/path.txt");
/// if f.exists() {
///     let reader = f.open_reader(Open);
///     let mut mem = [0u8, 8*64000];
///     reader.read(mem);
///     // ...
/// }
/// ```
///
/// * Is the given path a file?
///
/// ```rust
/// let f = get_file_path_from_wherever();
/// match f.is_file() {
///    true => doing_something_with_a_file(f),
///    _ => {}
/// }
/// ```
pub trait FileInfo : FileSystemInfo {
    /// Whether the underlying implemention (be it a file path,
    /// or something else) points at a "regular file" on the FS. Will return
    /// false for paths to non-existent locations or directories or
    /// other non-regular files (named pipes, etc).
    ///
    /// # Errors
    ///
    /// Will not raise a condition
    fn is_file(&self) -> bool {
        match ignore_io_error(|| self.stat()) {
            Some(s) => s.is_file,
            None => false
        }
    }

    /// Attempts to open a regular file for reading/writing based
    /// on provided inputs
    ///
    /// See `file::open` for more information on runtime semantics and error conditions
    fn open_stream(&self, mode: FileMode, access: FileAccess) -> Option<FileStream> {
        match ignore_io_error(|| self.stat()) {
            Some(s) => match s.is_file {
                true => open(self.get_path(), mode, access),
                false => None
            },
            None => open(self.get_path(), mode, access)
        }
    }

    /// Attempts to open a regular file in read-only mode, based
    /// on provided inputs
    ///
    /// See `file::open` for more information on runtime semantics and error conditions
    fn open_reader(&self, mode: FileMode) -> Option<FileReader> {
        match self.open_stream(mode, Read) {
            Some(s) => Some(FileReader { stream: s}),
            None => None
        }
    }

    /// Attempts to open a regular file in write-only mode, based
    /// on provided inputs
    ///
    /// See `file::open` for more information on runtime semantics and error conditions
    fn open_writer(&self, mode: FileMode) -> Option<FileWriter> {
        match self.open_stream(mode, Write) {
            Some(s) => Some(FileWriter { stream: s}),
            None => None
        }
    }

    /// Attempt to remove a file from the filesystem
    ///
    /// See `file::unlink` for more information on runtime semantics and error conditions
    fn unlink(&self) {
        unlink(self.get_path());
    }
}

/// `FileSystemInfo` implementation for `Path`s
impl FileSystemInfo for Path {
    fn get_path<'a>(&'a self) -> &'a Path { self }
}

/// `FileInfo` implementation for `Path`s
impl FileInfo for Path { }

/// Represents a directory, whose underlying path may or may not be valid
///
/// # Example
///
/// * Check if a directory exists, `mkdir`'ing it if not
///
/// ```rust
/// use std;
/// use std::path::Path;
/// use std::rt::io::file::{DirectoryInfo};
///
/// let dir = &Path("/some/dir");
/// if !dir.exists() {
///     dir.mkdir();
/// }
/// ```
///
/// * Is the given path a directory? If so, iterate on its contents
///
/// ```rust
/// fn visit_dirs(dir: &Path, cb: &fn(&Path)) {
///     if dir.is_dir() {
///         let contents = dir.readdir();
///         for entry in contents.iter() {
///             if entry.is_dir() { visit_dirs(entry, cb); }
///             else { cb(entry); }
///         }
///     }
///     else { fail2!("nope"); }
/// }
/// ```
pub trait DirectoryInfo : FileSystemInfo {
    /// Whether the underlying implemention (be it a file path,
    /// or something else) is pointing at a directory in the underlying FS.
    /// Will return false for paths to non-existent locations or if the item is
    /// not a directory (eg files, named pipes, links, etc)
    ///
    /// # Errors
    ///
    /// Will not raise a condition
    fn is_dir(&self) -> bool {
        match ignore_io_error(|| self.stat()) {
            Some(s) => s.is_dir,
            None => false
        }
    }

    /// Create a directory at the location pointed to by the
    /// type underlying the given `DirectoryInfo`.
    ///
    /// # Errors
    ///
    /// This method will raise a `PathAlreadyExists` kind of `io_error` condition
    /// if the provided path exists
    ///
    /// See `file::mkdir` for more information on runtime semantics and error conditions
    fn mkdir(&self) {
        match ignore_io_error(|| self.stat()) {
            Some(_) => {
                io_error::cond.raise(IoError {
                    kind: PathAlreadyExists,
                    desc: "Path already exists",
                    detail:
                        Some(format!("{} already exists; can't mkdir it",
                                     self.get_path().to_str()))
                })
            },
            None => mkdir(self.get_path())
        }
    }

    /// Remove a directory at the given location.
    ///
    /// # Errors
    ///
    /// This method will raise a `PathDoesntExist` kind of `io_error` condition
    /// if the provided path exists. It will raise a `MismatchedFileTypeForOperation`
    /// kind of `io_error` condition if the provided path points at any
    /// non-directory file type
    ///
    /// See `file::rmdir` for more information on runtime semantics and error conditions
    fn rmdir(&self) {
        match ignore_io_error(|| self.stat()) {
            Some(s) => {
                match s.is_dir {
                    true => rmdir(self.get_path()),
                    false => {
                        let ioerr = IoError {
                            kind: MismatchedFileTypeForOperation,
                            desc: "Cannot do rmdir() on a non-directory",
                            detail: Some(format!(
                                "{} is a non-directory; can't rmdir it",
                                self.get_path().to_str()))
                        };
                        io_error::cond.raise(ioerr);
                    }
                }
            },
            None =>
                io_error::cond.raise(IoError {
                    kind: PathDoesntExist,
                    desc: "Path doesn't exist",
                    detail: Some(format!("{} doesn't exist; can't rmdir it",
                                         self.get_path().to_str()))
                })
        }
    }

    // Get a collection of all entries at the given
    // directory
    fn readdir(&self) -> Option<~[Path]> {
        readdir(self.get_path())
    }
}

/// `DirectoryInfo` impl for `path::Path`
impl DirectoryInfo for Path { }

#[cfg(test)]
mod test {
    use super::super::{SeekSet, SeekCur, SeekEnd,
                       io_error, Read, Create, Open, ReadWrite};
    use super::super::super::test::*;
    use option::{Some, None};
    use path::Path;
    use super::*;
    use iter::range;
    #[test]
    fn file_test_io_smoke_test() {
        do run_in_mt_newsched_task {
            let message = "it's alright. have a good time";
            let filename = &Path("./tmp/file_rt_io_file_test.txt");
            {
                let mut write_stream = open(filename, Create, ReadWrite).unwrap();
                write_stream.write(message.as_bytes());
            }
            {
                use str;
                let mut read_stream = open(filename, Open, Read).unwrap();
                let mut read_buf = [0, .. 1028];
                let read_str = match read_stream.read(read_buf).unwrap() {
                    -1|0 => fail2!("shouldn't happen"),
                    n => str::from_utf8(read_buf.slice_to(n))
                };
                assert!(read_str == message.to_owned());
            }
            unlink(filename);
        }
    }

    #[test]
    fn file_test_io_invalid_path_opened_without_create_should_raise_condition() {
        do run_in_mt_newsched_task {
            let filename = &Path("./tmp/file_that_does_not_exist.txt");
            let mut called = false;
            do io_error::cond.trap(|_| {
                called = true;
            }).inside {
                let result = open(filename, Open, Read);
                assert!(result.is_none());
            }
            assert!(called);
        }
    }

    #[test]
    fn file_test_iounlinking_invalid_path_should_raise_condition() {
        do run_in_mt_newsched_task {
            let filename = &Path("./tmp/file_another_file_that_does_not_exist.txt");
            let mut called = false;
            do io_error::cond.trap(|_| {
                called = true;
            }).inside {
                unlink(filename);
            }
            assert!(called);
        }
    }

    #[test]
    fn file_test_io_non_positional_read() {
        do run_in_mt_newsched_task {
            use str;
            let message = "ten-four";
            let mut read_mem = [0, .. 8];
            let filename = &Path("./tmp/file_rt_io_file_test_positional.txt");
            {
                let mut rw_stream = open(filename, Create, ReadWrite).unwrap();
                rw_stream.write(message.as_bytes());
            }
            {
                let mut read_stream = open(filename, Open, Read).unwrap();
                {
                    let read_buf = read_mem.mut_slice(0, 4);
                    read_stream.read(read_buf);
                }
                {
                    let read_buf = read_mem.mut_slice(4, 8);
                    read_stream.read(read_buf);
                }
            }
            unlink(filename);
            let read_str = str::from_utf8(read_mem);
            assert!(read_str == message.to_owned());
        }
    }

    #[test]
    fn file_test_io_seek_and_tell_smoke_test() {
        do run_in_mt_newsched_task {
            use str;
            let message = "ten-four";
            let mut read_mem = [0, .. 4];
            let set_cursor = 4 as u64;
            let mut tell_pos_pre_read;
            let mut tell_pos_post_read;
            let filename = &Path("./tmp/file_rt_io_file_test_seeking.txt");
            {
                let mut rw_stream = open(filename, Create, ReadWrite).unwrap();
                rw_stream.write(message.as_bytes());
            }
            {
                let mut read_stream = open(filename, Open, Read).unwrap();
                read_stream.seek(set_cursor as i64, SeekSet);
                tell_pos_pre_read = read_stream.tell();
                read_stream.read(read_mem);
                tell_pos_post_read = read_stream.tell();
            }
            unlink(filename);
            let read_str = str::from_utf8(read_mem);
            assert!(read_str == message.slice(4, 8).to_owned());
            assert!(tell_pos_pre_read == set_cursor);
            assert!(tell_pos_post_read == message.len() as u64);
        }
    }

    #[test]
    fn file_test_io_seek_and_write() {
        do run_in_mt_newsched_task {
            use str;
            let initial_msg =   "food-is-yummy";
            let overwrite_msg =    "-the-bar!!";
            let final_msg =     "foo-the-bar!!";
            let seek_idx = 3;
            let mut read_mem = [0, .. 13];
            let filename = &Path("./tmp/file_rt_io_file_test_seek_and_write.txt");
            {
                let mut rw_stream = open(filename, Create, ReadWrite).unwrap();
                rw_stream.write(initial_msg.as_bytes());
                rw_stream.seek(seek_idx as i64, SeekSet);
                rw_stream.write(overwrite_msg.as_bytes());
            }
            {
                let mut read_stream = open(filename, Open, Read).unwrap();
                read_stream.read(read_mem);
            }
            unlink(filename);
            let read_str = str::from_utf8(read_mem);
            assert!(read_str == final_msg.to_owned());
        }
    }

    #[test]
    fn file_test_io_seek_shakedown() {
        do run_in_mt_newsched_task {
            use str;          // 01234567890123
            let initial_msg =   "qwer-asdf-zxcv";
            let chunk_one = "qwer";
            let chunk_two = "asdf";
            let chunk_three = "zxcv";
            let mut read_mem = [0, .. 4];
            let filename = &Path("./tmp/file_rt_io_file_test_seek_shakedown.txt");
            {
                let mut rw_stream = open(filename, Create, ReadWrite).unwrap();
                rw_stream.write(initial_msg.as_bytes());
            }
            {
                let mut read_stream = open(filename, Open, Read).unwrap();

                read_stream.seek(-4, SeekEnd);
                read_stream.read(read_mem);
                let read_str = str::from_utf8(read_mem);
                assert!(read_str == chunk_three.to_owned());

                read_stream.seek(-9, SeekCur);
                read_stream.read(read_mem);
                let read_str = str::from_utf8(read_mem);
                assert!(read_str == chunk_two.to_owned());

                read_stream.seek(0, SeekSet);
                read_stream.read(read_mem);
                let read_str = str::from_utf8(read_mem);
                assert!(read_str == chunk_one.to_owned());
            }
            unlink(filename);
        }
    }

    #[test]
    fn file_test_stat_is_correct_on_is_file() {
        do run_in_mt_newsched_task {
            let filename = &Path("./tmp/file_stat_correct_on_is_file.txt");
            {
                let mut fs = open(filename, Create, ReadWrite).unwrap();
                let msg = "hw";
                fs.write(msg.as_bytes());
            }
            let stat_res = match stat(filename) {
                Some(s) => s,
                None => fail2!("shouldn't happen")
            };
            assert!(stat_res.is_file);
            unlink(filename);
        }
    }

    #[test]
    fn file_test_stat_is_correct_on_is_dir() {
        do run_in_mt_newsched_task {
            let filename = &Path("./tmp/file_stat_correct_on_is_dir");
            mkdir(filename);
            let stat_res = match stat(filename) {
                Some(s) => s,
                None => fail2!("shouldn't happen")
            };
            assert!(stat_res.is_dir);
            rmdir(filename);
        }
    }

    #[test]
    fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
        do run_in_mt_newsched_task {
            let dir = &Path("./tmp/fileinfo_false_on_dir");
            mkdir(dir);
            assert!(dir.is_file() == false);
            rmdir(dir);
        }
    }

    #[test]
    fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
        do run_in_mt_newsched_task {
            let file = &Path("./tmp/fileinfo_check_exists_b_and_a.txt");
            {
                let msg = "foo".as_bytes();
                let mut w = file.open_writer(Create);
                w.write(msg);
            }
            assert!(file.exists());
            file.unlink();
            assert!(!file.exists());
        }
    }

    #[test]
    fn file_test_directoryinfo_check_exists_before_and_after_mkdir() {
        do run_in_mt_newsched_task {
            let dir = &Path("./tmp/before_and_after_dir");
            assert!(!dir.exists());
            dir.mkdir();
            assert!(dir.exists());
            assert!(dir.is_dir());
            dir.rmdir();
            assert!(!dir.exists());
        }
    }

    #[test]
    fn file_test_directoryinfo_readdir() {
        use str;
        do run_in_mt_newsched_task {
            let dir = &Path("./tmp/di_readdir");
            dir.mkdir();
            let prefix = "foo";
            for n in range(0,3) {
                let f = dir.push(format!("{}.txt", n));
                let mut w = f.open_writer(Create);
                let msg_str = (prefix + n.to_str().to_owned()).to_owned();
                let msg = msg_str.as_bytes();
                w.write(msg);
            }
            match dir.readdir() {
                Some(files) => {
                    let mut mem = [0u8, .. 4];
                    for f in files.iter() {
                        {
                            let n = f.filestem();
                            let mut r = f.open_reader(Open);
                            r.read(mem);
                            let read_str = str::from_utf8(mem);
                            let expected = match n {
                                Some(n) => prefix+n,
                                None => fail2!("really shouldn't happen..")
                            };
                            assert!(expected == read_str);
                        }
                        f.unlink();
                    }
                },
                None => fail2!("shouldn't happen")
            }
            dir.rmdir();
        }
    }
}
