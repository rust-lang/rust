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

At the top-level of the module are a set of freestanding functions, associated
with various filesystem operations. They all operate on a `Path` object.

All operations in this module, including those as part of `FileStream` et al
block the task during execution. Most will raise `std::rt::io::io_error`
conditions in the event of failure.

Also included in this module is an implementation block on the `Path` object
defined in `std::path::Path`. The impl adds useful methods about inspecting the
metadata of a file. This includes getting the `stat` information, reading off
particular bits of it, etc.

*/

use c_str::ToCStr;
use super::{Reader, Writer, Seek};
use super::{SeekStyle, Read, Write, Open, CreateOrTruncate,
            FileMode, FileAccess, FileStat, io_error, FilePermission};
use rt::rtio::{RtioFileStream, IoFactory, with_local_io};
use rt::io;
use option::{Some, None, Option};
use result::{Ok, Err};
use path;
use path::{Path, GenericPath};

/// Open a file for reading/writing, as indicated by `path`.
///
/// # Example
///
///     use std::rt::{io, file, io_error};
///
///     let p = Path::new("/some/file/path.txt");
///
///     do io_error::cond.trap(|_| {
///         // hoo-boy...
///     }).inside {
///         let stream = match file::open_stream(&p, io::Create, io::ReadWrite) {
///             Some(s) => s,
///             None => fail!("whoops! I'm sure this raised, anyways..")
///         };
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
/// `FileStream` opened as `ReadOnly` will raise an `io_error` condition at
/// runtime). If you desire a more-correctly-constrained interface to files,
/// use the `{open_stream, open, create}` methods that are a part of `Path`.
///
/// # Errors
///
/// This function will raise an `io_error` condition under a number of different
/// circumstances, to include but not limited to:
///
/// * Opening a file that already exists with `FileMode` of `Create` or vice
///   versa (e.g.  opening a non-existant file with `FileMode` or `Open`)
/// * Attempting to open a file with a `FileAccess` that the user lacks
///   permissions for
/// * Filesystem-level errors (full disk, etc)
pub fn open_stream(path: &Path,
                   mode: FileMode,
                   access: FileAccess) -> Option<FileStream> {
    do with_local_io |io| {
        match io.fs_open(&path.to_c_str(), mode, access) {
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
}

/// Attempts to open a file in read-only mode. This function is equivalent to
/// `open_stream(path, Open, Read)`, and will raise all of the same errors that
/// `open_stream` does.
///
/// For more information, see the `open_stream` function.
pub fn open(path: &Path) -> Option<FileReader> {
    open_stream(path, Open, Read).map(|s| FileReader { stream: s })
}

/// Attempts to create a file in write-only mode. This function is equivalent to
/// `open_stream(path, CreateOrTruncate, Write)`, and will raise all of the
/// same errors that `open_stream` does.
///
/// For more information, see the `open_stream` function.
pub fn create(path: &Path) -> Option<FileWriter> {
    open_stream(path, CreateOrTruncate, Write).map(|s| FileWriter { stream: s })
}

/// Unlink a file from the underlying filesystem.
///
/// # Example
///
///     use std::rt::io::file;
///
///     let p = Path::new("/some/file/path.txt");
///     file::unlink(&p);
///     // if we made it here without failing, then the
///     // unlink operation was successful
///
/// Note that, just because an unlink call was successful, it is not
/// guaranteed that a file is immediately deleted (e.g. depending on
/// platform, other open file descriptors may prevent immediate removal)
///
/// # Errors
///
/// This function will raise an `io_error` condition if the path points to a
/// directory, the user lacks permissions to remove the file, or if some other
/// filesystem-level error occurs.
pub fn unlink(path: &Path) {
    do with_local_io |io| {
        match io.fs_unlink(&path.to_c_str()) {
            Ok(()) => Some(()),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    };
}

/// Create a new, empty directory at the provided path
///
/// # Example
///
///     use std::libc::S_IRWXU;
///     use std::rt::io::file;
///
///     let p = Path::new("/some/dir");
///     file::mkdir(&p, S_IRWXU as int);
///     // If we got here, our directory exists! Horray!
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks permissions
/// to make a new directory at the provided path, or if the directory already
/// exists.
pub fn mkdir(path: &Path, mode: FilePermission) {
    do with_local_io |io| {
        match io.fs_mkdir(&path.to_c_str(), mode) {
            Ok(_) => Some(()),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    };
}

/// Remove an existing, empty directory
///
/// # Example
///
///     use std::rt::io::file;
///
///     let p = Path::new("/some/dir");
///     file::rmdir(&p);
///     // good riddance, you mean ol' directory
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks permissions
/// to remove the directory at the provided path, or if the directory isn't
/// empty.
pub fn rmdir(path: &Path) {
    do with_local_io |io| {
        match io.fs_rmdir(&path.to_c_str()) {
            Ok(_) => Some(()),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    };
}

/// Get information on the file, directory, etc at the provided path
///
/// Given a path, query the file system to get information about a file,
/// directory, etc.
///
/// Returns a fully-filled out stat structure on succes, and on failure it will
/// return a dummy stat structure (it is expected that the condition raised is
/// handled as well).
///
/// # Example
///
///     use std::rt::io::{file, io_error};
///
///     let p = Path::new("/some/file/path.txt");
///
///     do io_error::cond.trap(|_| {
///         // hoo-boy...
///     }).inside {
///         let info = file::stat(p);
///         if info.is_file {
///             // just imagine the possibilities ...
///         }
///     }
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks the requisite
/// permissions to perform a `stat` call on the given path or if there is no
/// entry in the filesystem at the provided path.
pub fn stat(path: &Path) -> FileStat {
    do with_local_io |io| {
        match io.fs_stat(&path.to_c_str()) {
            Ok(p) => Some(p),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    }.unwrap_or_else(|| {
        FileStat {
            path: Path::new(path.to_c_str()),
            is_file: false,
            is_dir: false,
            device: 0,
            mode: 0,
            inode: 0,
            size: 0,
            created: 0,
            modified: 0,
            accessed: 0,
        }
    })
}

/// Retrieve a vector containing all entries within a provided directory
///
/// # Example
///
///     use std::rt::io::file;
///
///     fn visit_dirs(dir: &Path, cb: &fn(&Path)) {
///         if dir.is_dir() {
///             let contents = file::readdir(dir).unwrap();
///             for entry in contents.iter() {
///                 if entry.is_dir() { visit_dirs(entry, cb); }
///                 else { cb(entry); }
///             }
///         }
///         else { fail!("nope"); }
///     }
///
/// # Errors
///
/// Will raise an `io_error` condition if the provided `from` doesn't exist,
/// the process lacks permissions to view the contents or if the `path` points
/// at a non-directory file
pub fn readdir(path: &Path) -> ~[Path] {
    do with_local_io |io| {
        match io.fs_readdir(&path.to_c_str(), 0) {
            Ok(p) => Some(p),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    }.unwrap_or_else(|| ~[])
}

/// Rename a file or directory to a new name.
///
/// # Example
///
///     use std::rt::io::file;
///
///     file::rename(Path::new("foo"), Path::new("bar"));
///     // Oh boy, nothing was raised!
///
/// # Errors
///
/// Will raise an `io_error` condition if the provided `path` doesn't exist,
/// the process lacks permissions to view the contents, or if some other
/// intermittent I/O error occurs.
pub fn rename(from: &Path, to: &Path) {
    do with_local_io |io| {
        match io.fs_rename(&from.to_c_str(), &to.to_c_str()) {
            Ok(()) => Some(()),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    };
}

/// Copies the contents of one file to another.
///
/// # Example
///
///     use std::rt::io::file;
///
///     file::copy(Path::new("foo.txt"), Path::new("bar.txt"));
///     // Oh boy, nothing was raised!
///
/// # Errors
///
/// Will raise an `io_error` condition if the provided `from` doesn't exist,
/// the process lacks permissions to view the contents, or if some other
/// intermittent I/O error occurs (such as `to` couldn't be created).
pub fn copy(from: &Path, to: &Path) {
    let mut reader = match open(from) { Some(f) => f, None => return };
    let mut writer = match create(to) { Some(f) => f, None => return };
    let mut buf = [0, ..io::DEFAULT_BUF_SIZE];

    loop {
        match reader.read(buf) {
            Some(amt) => writer.write(buf.slice_to(amt)),
            None => break
        }
    }

    // FIXME(#10131) this is an awful way to pull out the permission bits.
    //               If this comment is removed, then there should be a test
    //               asserting that permission bits are maintained using the
    //               permission interface created.
    chmod(to, (from.stat().mode & 0xfff) as u32);
}

// This function is not public because it's got a terrible interface for `mode`
// FIXME(#10131)
fn chmod(path: &Path, mode: u32) {
    do with_local_io |io| {
        match io.fs_chmod(&path.to_c_str(), mode) {
            Ok(()) => Some(()),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    };
}

/// Recursively walk a directory structure. This function will call the
/// provided closure on all directories and files found inside the path
/// pointed to by `self`. If the closure returns `false`, then the iteration
/// will be short-circuited.
pub fn walk_dir(path: &Path, f: &fn(&Path) -> bool) -> bool {
    let files = readdir(path);
    files.iter().advance(|q| {
        f(q) && (!q.is_dir() || walk_dir(q, |p| f(p)))
    })
}

/// Recursively create a directory and all of its parent components if they
/// are missing.
///
/// # Failure
///
/// This function will raise on the `io_error` condition if an error
/// happens, see `file::mkdir` for more information about error conditions
/// and performance.
pub fn mkdir_recursive(path: &Path, mode: FilePermission) {
    // tjc: if directory exists but with different permissions,
    // should we return false?
    if path.is_dir() {
        return
    }
    if path.filename().is_some() {
        mkdir_recursive(&path.dir_path(), mode);
    }
    mkdir(path, mode)
}

/// Removes a directory at this path, after removing all its contents. Use
/// carefully!
///
/// # Failure
///
/// This function will raise on the `io_error` condition if an error
/// happens. See `file::unlink` and `file::readdir` for possible error
/// conditions.
pub fn rmdir_recursive(path: &Path) {
    do walk_dir(path) |inner| {
        if inner.is_dir() {
            rmdir_recursive(inner);
        } else {
            unlink(inner);
        }
        true
    };
    // Directory should now be empty
    rmdir(path);
}

/// Constrained version of `FileStream` that only exposes read-specific
/// operations.
///
/// Can be retreived via `Path.open()` or `file::open`.
pub struct FileReader { priv stream: FileStream }

impl Reader for FileReader {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.stream.read(buf) }
    fn eof(&mut self) -> bool { self.stream.eof() }
}

impl Seek for FileReader {
    fn tell(&self) -> u64 { self.stream.tell() }
    fn seek(&mut self, p: i64, s: SeekStyle) { self.stream.seek(p, s) }
}

/// Constrained version of `FileStream` that only exposes write-specific
/// operations.
///
/// Can be retreived via `Path.create()` or `file::create`.
pub struct FileWriter { priv stream: FileStream }

impl Writer for FileWriter {
    fn write(&mut self, buf: &[u8]) { self.stream.write(buf); }
    fn flush(&mut self) { self.stream.flush(); }
}

impl Seek for FileWriter {
    fn tell(&self) -> u64 { self.stream.tell() }
    fn seek(&mut self, p: i64, s: SeekStyle) { self.stream.seek(p, s); }
}

/// Unconstrained file access type that exposes read and write operations
///
/// Can be retreived via `file::open()` and `Path.open_stream()`.
///
/// # Errors
///
/// This type will raise an io_error condition if operations are attempted against
/// it for which its underlying file descriptor was not configured at creation
/// time, via the `FileAccess` parameter to `file::open()`.
///
/// For this reason, it is best to use the access-constrained wrappers that are
/// exposed via `Path.open()` and `Path.create()`.
pub struct FileStream {
    priv fd: ~RtioFileStream,
    priv last_nread: int,
}

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
                if ioerr.kind != io::EndOfFile {
                    io_error::cond.raise(ioerr);
                }
                return None;
            }
        }
    }

    fn eof(&mut self) -> bool { self.last_nread == 0 }
}

impl Writer for FileStream {
    fn write(&mut self, buf: &[u8]) {
        match self.fd.write(buf) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }
}

impl Seek for FileStream {
    fn tell(&self) -> u64 {
        let res = self.fd.tell();
        match res {
            Ok(cursor) => cursor,
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
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
                io_error::cond.raise(ioerr);
            }
        }
    }
}

impl path::Path {
    /// Get information on the file, directory, etc at this path.
    ///
    /// Consult the `file::stat` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with `file::stat`.
    pub fn stat(&self) -> FileStat { stat(self) }

    /// Boolean value indicator whether the underlying file exists on the local
    /// filesystem. This will return true if the path points to either a
    /// directory or a file.
    ///
    /// # Errors
    ///
    /// Will not raise a condition
    pub fn exists(&self) -> bool {
        io::result(|| self.stat()).is_ok()
    }

    /// Whether the underlying implemention (be it a file path, or something
    /// else) points at a "regular file" on the FS. Will return false for paths
    /// to non-existent locations or directories or other non-regular files
    /// (named pipes, etc).
    ///
    /// # Errors
    ///
    /// Will not raise a condition
    pub fn is_file(&self) -> bool {
        match io::result(|| self.stat()) {
            Ok(s) => s.is_file,
            Err(*) => false
        }
    }

    /// Whether the underlying implemention (be it a file path,
    /// or something else) is pointing at a directory in the underlying FS.
    /// Will return false for paths to non-existent locations or if the item is
    /// not a directory (eg files, named pipes, links, etc)
    ///
    /// # Errors
    ///
    /// Will not raise a condition
    pub fn is_dir(&self) -> bool {
        match io::result(|| self.stat()) {
            Ok(s) => s.is_dir,
            Err(*) => false
        }
    }
}

#[cfg(test)]
mod test {
    use path::{Path, GenericPath};
    use result::{Ok, Err};
    use option::{Some, None};
    use iter::range;
    use rt::test::run_in_mt_newsched_task;
    use super::{open_stream, unlink, stat, copy, rmdir, mkdir, readdir,
                open, create, rmdir_recursive, mkdir_recursive};

    use rt::io;
    use rt::io::Reader;
    use super::super::{SeekSet, SeekCur, SeekEnd,
                       io_error, Read, Create, Open, ReadWrite};
    use vec::Vector;

    fn tmpdir() -> Path {
        use os;
        use rand;
        let ret = os::tmpdir().join(format!("rust-{}", rand::random::<u32>()));
        mkdir(&ret, io::UserRWX);
        ret
    }

    #[test]
    fn file_test_io_smoke_test() {
        do run_in_mt_newsched_task {
            let message = "it's alright. have a good time";
            let filename = &Path::new("./tmp/file_rt_io_file_test.txt");
            {
                let mut write_stream = open_stream(filename, Create, ReadWrite);
                write_stream.write(message.as_bytes());
            }
            {
                use str;
                let mut read_stream = open_stream(filename, Open, Read);
                let mut read_buf = [0, .. 1028];
                let read_str = match read_stream.read(read_buf).unwrap() {
                    -1|0 => fail!("shouldn't happen"),
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
            let filename = &Path::new("./tmp/file_that_does_not_exist.txt");
            let mut called = false;
            do io_error::cond.trap(|_| {
                called = true;
            }).inside {
                let result = open_stream(filename, Open, Read);
                assert!(result.is_none());
            }
            assert!(called);
        }
    }

    #[test]
    fn file_test_iounlinking_invalid_path_should_raise_condition() {
        do run_in_mt_newsched_task {
            let filename = &Path::new("./tmp/file_another_file_that_does_not_exist.txt");
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
            let filename = &Path::new("./tmp/file_rt_io_file_test_positional.txt");
            {
                let mut rw_stream = open_stream(filename, Create, ReadWrite);
                rw_stream.write(message.as_bytes());
            }
            {
                let mut read_stream = open_stream(filename, Open, Read);
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
            let filename = &Path::new("./tmp/file_rt_io_file_test_seeking.txt");
            {
                let mut rw_stream = open_stream(filename, Create, ReadWrite);
                rw_stream.write(message.as_bytes());
            }
            {
                let mut read_stream = open_stream(filename, Open, Read);
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
            let filename = &Path::new("./tmp/file_rt_io_file_test_seek_and_write.txt");
            {
                let mut rw_stream = open_stream(filename, Create, ReadWrite);
                rw_stream.write(initial_msg.as_bytes());
                rw_stream.seek(seek_idx as i64, SeekSet);
                rw_stream.write(overwrite_msg.as_bytes());
            }
            {
                let mut read_stream = open_stream(filename, Open, Read);
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
            let filename = &Path::new("./tmp/file_rt_io_file_test_seek_shakedown.txt");
            {
                let mut rw_stream = open_stream(filename, Create, ReadWrite);
                rw_stream.write(initial_msg.as_bytes());
            }
            {
                let mut read_stream = open_stream(filename, Open, Read);

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
            let filename = &Path::new("./tmp/file_stat_correct_on_is_file.txt");
            {
                let mut fs = open_stream(filename, Create, ReadWrite);
                let msg = "hw";
                fs.write(msg.as_bytes());
            }
            let stat_res = stat(filename);
            assert!(stat_res.is_file);
            unlink(filename);
        }
    }

    #[test]
    fn file_test_stat_is_correct_on_is_dir() {
        do run_in_mt_newsched_task {
            let filename = &Path::new("./tmp/file_stat_correct_on_is_dir");
            mkdir(filename, io::UserRWX);
            let stat_res = filename.stat();
            assert!(stat_res.is_dir);
            rmdir(filename);
        }
    }

    #[test]
    fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
        do run_in_mt_newsched_task {
            let dir = &Path::new("./tmp/fileinfo_false_on_dir");
            mkdir(dir, io::UserRWX);
            assert!(dir.is_file() == false);
            rmdir(dir);
        }
    }

    #[test]
    fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
        do run_in_mt_newsched_task {
            let file = &Path::new("./tmp/fileinfo_check_exists_b_and_a.txt");
            create(file).write(bytes!("foo"));
            assert!(file.exists());
            unlink(file);
            assert!(!file.exists());
        }
    }

    #[test]
    fn file_test_directoryinfo_check_exists_before_and_after_mkdir() {
        do run_in_mt_newsched_task {
            let dir = &Path::new("./tmp/before_and_after_dir");
            assert!(!dir.exists());
            mkdir(dir, io::UserRWX);
            assert!(dir.exists());
            assert!(dir.is_dir());
            rmdir(dir);
            assert!(!dir.exists());
        }
    }

    #[test]
    fn file_test_directoryinfo_readdir() {
        use str;
        do run_in_mt_newsched_task {
            let dir = &Path::new("./tmp/di_readdir");
            mkdir(dir, io::UserRWX);
            let prefix = "foo";
            for n in range(0,3) {
                let f = dir.join(format!("{}.txt", n));
                let mut w = create(&f);
                let msg_str = (prefix + n.to_str().to_owned()).to_owned();
                let msg = msg_str.as_bytes();
                w.write(msg);
            }
            let files = readdir(dir);
            let mut mem = [0u8, .. 4];
            for f in files.iter() {
                {
                    let n = f.filestem_str();
                    open(f).read(mem);
                    let read_str = str::from_utf8(mem);
                    let expected = match n {
                        None|Some("") => fail!("really shouldn't happen.."),
                        Some(n) => prefix+n
                    };
                    assert!(expected == read_str);
                }
                unlink(f);
            }
            rmdir(dir);
        }
    }

    #[test]
    fn recursive_mkdir_slash() {
        mkdir_recursive(&Path::new("/"), io::UserRWX);
    }

    #[test]
    fn unicode_path_is_dir() {
        assert!(Path::new(".").is_dir());
        assert!(!Path::new("test/stdtest/fs.rs").is_dir());

        let tmpdir = tmpdir();

        let mut dirpath = tmpdir.clone();
        dirpath.push(format!("test-가一ー你好"));
        mkdir(&dirpath, io::UserRWX);
        assert!(dirpath.is_dir());

        let mut filepath = dirpath;
        filepath.push("unicode-file-\uac00\u4e00\u30fc\u4f60\u597d.rs");
        create(&filepath); // ignore return; touch only
        assert!(!filepath.is_dir());
        assert!(filepath.exists());

        rmdir_recursive(&tmpdir);
    }

    #[test]
    fn unicode_path_exists() {
        assert!(Path::new(".").exists());
        assert!(!Path::new("test/nonexistent-bogus-path").exists());

        let tmpdir = tmpdir();
        let unicode = tmpdir.clone();
        let unicode = unicode.join(format!("test-각丁ー再见"));
        mkdir(&unicode, io::UserRWX);
        assert!(unicode.exists());
        assert!(!Path::new("test/unicode-bogus-path-각丁ー再见").exists());
        rmdir_recursive(&tmpdir);
    }

    #[test]
    fn copy_file_does_not_exist() {
        let from = Path::new("test/nonexistent-bogus-path");
        let to = Path::new("test/other-bogus-path");
        match io::result(|| copy(&from, &to)) {
            Ok(*) => fail!(),
            Err(*) => {
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

        create(&input).write(bytes!("hello"));
        copy(&input, &out);
        let contents = open(&out).read_to_end();
        assert_eq!(contents.as_slice(), bytes!("hello"));

        assert_eq!(input.stat().mode, out.stat().mode);
        rmdir_recursive(&tmpdir);
    }
}
