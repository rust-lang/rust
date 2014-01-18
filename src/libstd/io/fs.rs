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

All operations in this module, including those as part of `File` et al
block the task during execution. Most will raise `std::io::io_error`
conditions in the event of failure.

Also included in this module is an implementation block on the `Path` object
defined in `std::path::Path`. The impl adds useful methods about inspecting the
metadata of a file. This includes getting the `stat` information, reading off
particular bits of it, etc.

# Example

    use std::io::{File, fs};

    let path = Path::new("foo.txt");

    // create the file, whether it exists or not
    let mut file = File::create(&path);
    file.write(bytes!("foobar"));

    // open the file in read-only mode
    let mut file = File::open(&path);
    file.read_to_end();

    println!("{}", path.stat().size);
    fs::symlink(&path, &Path::new("bar.txt"));
    fs::unlink(&path);

*/

use c_str::ToCStr;
use clone::Clone;
use iter::Iterator;
use super::{Reader, Writer, Seek};
use super::{SeekStyle, Read, Write, Open, IoError, Truncate,
            FileMode, FileAccess, FileStat, io_error, FilePermission};
use rt::rtio::{RtioFileStream, IoFactory, LocalIo};
use io;
use option::{Some, None, Option};
use result::{Ok, Err};
use path;
use path::{Path, GenericPath};
use vec::{OwnedVector, ImmutableVector};

/// Unconstrained file access type that exposes read and write operations
///
/// Can be constructed via `File::open()`, `File::create()`, and
/// `File::open_mode()`.
///
/// # Errors
///
/// This type will raise an io_error condition if operations are attempted against
/// it for which its underlying file descriptor was not configured at creation
/// time, via the `FileAccess` parameter to `File::open_mode()`.
pub struct File {
    priv fd: ~RtioFileStream,
    priv path: Path,
    priv last_nread: int,
}

impl File {
    /// Open a file at `path` in the mode specified by the `mode` and `access`
    /// arguments
    ///
    /// # Example
    ///
    ///     use std::io::{File, io_error, Open, ReadWrite};
    ///
    ///     let p = Path::new("/some/file/path.txt");
    ///
    ///     io_error::cond.trap(|_| {
    ///         // hoo-boy...
    ///     }).inside(|| {
    ///         let file = match File::open_mode(&p, Open, ReadWrite) {
    ///             Some(s) => s,
    ///             None => fail!("whoops! I'm sure this raised, anyways..")
    ///         };
    ///         // do some stuff with that file
    ///
    ///         // the file will be closed at the end of this block
    ///     })
    ///     // ..
    ///
    /// `FileMode` and `FileAccess` provide information about the permissions
    /// context in which a given stream is created. More information about them
    /// can be found in `std::io`'s docs. If a file is opened with `Write`
    /// or `ReadWrite` access, then it will be created it it does not already
    /// exist.
    ///
    /// Note that, with this function, a `File` is returned regardless of the
    /// access-limitations indicated by `FileAccess` (e.g. calling `write` on a
    /// `File` opened as `Read` will raise an `io_error` condition at runtime).
    ///
    /// # Errors
    ///
    /// This function will raise an `io_error` condition under a number of
    /// different circumstances, to include but not limited to:
    ///
    /// * Opening a file that does not exist with `Read` access.
    /// * Attempting to open a file with a `FileAccess` that the user lacks
    ///   permissions for
    /// * Filesystem-level errors (full disk, etc)
    pub fn open_mode(path: &Path,
                     mode: FileMode,
                     access: FileAccess) -> Option<File> {
        LocalIo::maybe_raise(|io| {
            io.fs_open(&path.to_c_str(), mode, access).map(|fd| {
                File {
                    path: path.clone(),
                    fd: fd,
                    last_nread: -1
                }
            })
        })
    }

    /// Attempts to open a file in read-only mode. This function is equivalent to
    /// `File::open_mode(path, Open, Read)`, and will raise all of the same
    /// errors that `File::open_mode` does.
    ///
    /// For more information, see the `File::open_mode` function.
    ///
    /// # Example
    ///
    ///     use std::io::File;
    ///
    ///     let contents = File::open(&Path::new("foo.txt")).read_to_end();
    pub fn open(path: &Path) -> Option<File> {
        File::open_mode(path, Open, Read)
    }

    /// Attempts to create a file in write-only mode. This function is
    /// equivalent to `File::open_mode(path, Truncate, Write)`, and will
    /// raise all of the same errors that `File::open_mode` does.
    ///
    /// For more information, see the `File::open_mode` function.
    ///
    /// # Example
    ///
    ///     use std::io::File;
    ///
    ///     let mut f = File::create(&Path::new("foo.txt"));
    ///     f.write(bytes!("This is a sample file"));
    pub fn create(path: &Path) -> Option<File> {
        File::open_mode(path, Truncate, Write)
    }

    /// Returns the original path which was used to open this file.
    pub fn path<'a>(&'a self) -> &'a Path {
        &self.path
    }

    /// Synchronizes all modifications to this file to its permanent storage
    /// device. This will flush any internal buffers necessary to perform this
    /// operation.
    ///
    /// # Errors
    ///
    /// This function will raise on the `io_error` condition on failure.
    pub fn fsync(&mut self) {
        self.fd.fsync().map_err(|e| io_error::cond.raise(e));
    }

    /// This function is similar to `fsync`, except that it may not synchronize
    /// file metadata to the filesystem. This is intended for use case which
    /// must synchronize content, but don't need the metadata on disk. The goal
    /// of this method is to reduce disk operations.
    ///
    /// # Errors
    ///
    /// This function will raise on the `io_error` condition on failure.
    pub fn datasync(&mut self) {
        self.fd.datasync().map_err(|e| io_error::cond.raise(e));
    }

    /// Either truncates or extends the underlying file, updating the size of
    /// this file to become `size`. This is equivalent to unix's `truncate`
    /// function.
    ///
    /// If the `size` is less than the current file's size, then the file will
    /// be shrunk. If it is greater than the current file's size, then the file
    /// will be extended to `size` and have all of the intermediate data filled
    /// in with 0s.
    ///
    /// # Errors
    ///
    /// On error, this function will raise on the `io_error` condition.
    pub fn truncate(&mut self, size: i64) {
        self.fd.truncate(size).map_err(|e| io_error::cond.raise(e));
    }

    /// Tests whether this stream has reached EOF.
    ///
    /// If true, then this file will no longer continue to return data via
    /// `read`.
    pub fn eof(&self) -> bool {
        self.last_nread == 0
    }
}

/// Unlink a file from the underlying filesystem.
///
/// # Example
///
///     use std::io::fs;
///
///     let p = Path::new("/some/file/path.txt");
///     fs::unlink(&p);
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
/// directory, the user lacks permissions to remove the file, or if some
/// other filesystem-level error occurs.
pub fn unlink(path: &Path) {
    LocalIo::maybe_raise(|io| io.fs_unlink(&path.to_c_str()));
}

/// Given a path, query the file system to get information about a file,
/// directory, etc. This function will traverse symlinks to query
/// information about the destination file.
///
/// Returns a fully-filled out stat structure on success, and on failure it
/// will return a dummy stat structure (it is expected that the condition
/// raised is handled as well).
///
/// # Example
///
///     use std::io;
///     use std::io::fs;
///
///     let p = Path::new("/some/file/path.txt");
///     match io::result(|| fs::stat(&p)) {
///         Ok(stat) => { /* ... */ }
///         Err(e) => { /* handle error */ }
///     }
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks the
/// requisite permissions to perform a `stat` call on the given path or if
/// there is no entry in the filesystem at the provided path.
pub fn stat(path: &Path) -> FileStat {
    LocalIo::maybe_raise(|io| {
        io.fs_stat(&path.to_c_str())
    }).unwrap_or_else(dummystat)
}

fn dummystat() -> FileStat {
    FileStat {
        path: Path::new(""),
        size: 0,
        kind: io::TypeFile,
        perm: 0,
        created: 0,
        modified: 0,
        accessed: 0,
        unstable: io::UnstableFileStat {
            device: 0,
            inode: 0,
            rdev: 0,
            nlink: 0,
            uid: 0,
            gid: 0,
            blksize: 0,
            blocks: 0,
            flags: 0,
            gen: 0,
        }
    }
}

/// Perform the same operation as the `stat` function, except that this
/// function does not traverse through symlinks. This will return
/// information about the symlink file instead of the file that it points
/// to.
///
/// # Errors
///
/// See `stat`
pub fn lstat(path: &Path) -> FileStat {
    LocalIo::maybe_raise(|io| {
        io.fs_lstat(&path.to_c_str())
    }).unwrap_or_else(dummystat)
}

/// Rename a file or directory to a new name.
///
/// # Example
///
///     use std::io::fs;
///
///     fs::rename(&Path::new("foo"), &Path::new("bar"));
///     // Oh boy, nothing was raised!
///
/// # Errors
///
/// Will raise an `io_error` condition if the provided `path` doesn't exist,
/// the process lacks permissions to view the contents, or if some other
/// intermittent I/O error occurs.
pub fn rename(from: &Path, to: &Path) {
    LocalIo::maybe_raise(|io| io.fs_rename(&from.to_c_str(), &to.to_c_str()));
}

/// Copies the contents of one file to another. This function will also
/// copy the permission bits of the original file to the destination file.
///
/// Note that if `from` and `to` both point to the same file, then the file
/// will likely get truncated by this operation.
///
/// # Example
///
///     use std::io::fs;
///
///     fs::copy(&Path::new("foo.txt"), &Path::new("bar.txt"));
///     // Oh boy, nothing was raised!
///
/// # Errors
///
/// Will raise an `io_error` condition is the following situations, but is
/// not limited to just these cases:
///
/// * The `from` path is not a file
/// * The `from` file does not exist
/// * The current process does not have the permission rights to access
///   `from` or write `to`
///
/// Note that this copy is not atomic in that once the destination is
/// ensured to not exist, there is nothing preventing the destination from
/// being created and then destroyed by this operation.
pub fn copy(from: &Path, to: &Path) {
    if !from.is_file() {
        return io_error::cond.raise(IoError {
            kind: io::MismatchedFileTypeForOperation,
            desc: "the source path is not an existing file",
            detail: None,
        });
    }

    let mut reader = match File::open(from) { Some(f) => f, None => return };
    let mut writer = match File::create(to) { Some(f) => f, None => return };
    let mut buf = [0, ..io::DEFAULT_BUF_SIZE];

    loop {
        match reader.read(buf) {
            Some(amt) => writer.write(buf.slice_to(amt)),
            None => break
        }
    }

    chmod(to, from.stat().perm)
}

/// Changes the permission mode bits found on a file or a directory. This
/// function takes a mask from the `io` module
///
/// # Example
///
///     use std::io;
///     use std::io::fs;
///
///     fs::chmod(&Path::new("file.txt"), io::UserFile);
///     fs::chmod(&Path::new("file.txt"), io::UserRead | io::UserWrite);
///     fs::chmod(&Path::new("dir"),      io::UserDir);
///     fs::chmod(&Path::new("file.exe"), io::UserExec);
///
/// # Errors
///
/// If this function encounters an I/O error, it will raise on the `io_error`
/// condition. Some possible error situations are not having the permission to
/// change the attributes of a file or the file not existing.
pub fn chmod(path: &Path, mode: io::FilePermission) {
    LocalIo::maybe_raise(|io| io.fs_chmod(&path.to_c_str(), mode));
}

/// Change the user and group owners of a file at the specified path.
///
/// # Errors
///
/// This function will raise on the `io_error` condition on failure.
pub fn chown(path: &Path, uid: int, gid: int) {
    LocalIo::maybe_raise(|io| io.fs_chown(&path.to_c_str(), uid, gid));
}

/// Creates a new hard link on the filesystem. The `dst` path will be a
/// link pointing to the `src` path. Note that systems often require these
/// two paths to both be located on the same filesystem.
///
/// # Errors
///
/// This function will raise on the `io_error` condition on failure.
pub fn link(src: &Path, dst: &Path) {
    LocalIo::maybe_raise(|io| io.fs_link(&src.to_c_str(), &dst.to_c_str()));
}

/// Creates a new symbolic link on the filesystem. The `dst` path will be a
/// symlink pointing to the `src` path.
///
/// # Errors
///
/// This function will raise on the `io_error` condition on failure.
pub fn symlink(src: &Path, dst: &Path) {
    LocalIo::maybe_raise(|io| io.fs_symlink(&src.to_c_str(), &dst.to_c_str()));
}

/// Reads a symlink, returning the file that the symlink points to.
///
/// # Errors
///
/// This function will raise on the `io_error` condition on failure. Failure
/// conditions include reading a file that does not exist or reading a file
/// which is not a symlink.
pub fn readlink(path: &Path) -> Option<Path> {
    LocalIo::maybe_raise(|io| io.fs_readlink(&path.to_c_str()))
}

/// Create a new, empty directory at the provided path
///
/// # Example
///
///     use std::libc::S_IRWXU;
///     use std::io::fs;
///
///     let p = Path::new("/some/dir");
///     fs::mkdir(&p, S_IRWXU as int);
///     // If we got here, our directory exists! Hooray!
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks permissions
/// to make a new directory at the provided path, or if the directory already
/// exists.
pub fn mkdir(path: &Path, mode: FilePermission) {
    LocalIo::maybe_raise(|io| io.fs_mkdir(&path.to_c_str(), mode));
}

/// Remove an existing, empty directory
///
/// # Example
///
///     use std::io::fs;
///
///     let p = Path::new("/some/dir");
///     fs::rmdir(&p);
///     // good riddance, you mean ol' directory
///
/// # Errors
///
/// This call will raise an `io_error` condition if the user lacks permissions
/// to remove the directory at the provided path, or if the directory isn't
/// empty.
pub fn rmdir(path: &Path) {
    LocalIo::maybe_raise(|io| io.fs_rmdir(&path.to_c_str()));
}

/// Retrieve a vector containing all entries within a provided directory
///
/// # Example
///
///     use std::io::fs;
///
///     // one possible implementation of fs::walk_dir only visiting files
///     fn visit_dirs(dir: &Path, cb: |&Path|) {
///         if dir.is_dir() {
///             let contents = fs::readdir(dir).unwrap();
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
    LocalIo::maybe_raise(|io| {
        io.fs_readdir(&path.to_c_str(), 0)
    }).unwrap_or_else(|| ~[])
}

/// Returns an iterator which will recursively walk the directory structure
/// rooted at `path`. The path given will not be iterated over, and this will
/// perform iteration in a top-down order.
pub fn walk_dir(path: &Path) -> Directories {
    Directories { stack: readdir(path) }
}

/// An iterator which walks over a directory
pub struct Directories {
    priv stack: ~[Path],
}

impl Iterator<Path> for Directories {
    fn next(&mut self) -> Option<Path> {
        match self.stack.shift_opt() {
            Some(path) => {
                if path.is_dir() {
                    self.stack.push_all_move(readdir(&path));
                }
                Some(path)
            }
            None => None
        }
    }
}

/// Recursively create a directory and all of its parent components if they
/// are missing.
///
/// # Errors
///
/// This function will raise on the `io_error` condition if an error
/// happens, see `fs::mkdir` for more information about error conditions
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
/// # Errors
///
/// This function will raise on the `io_error` condition if an error
/// happens. See `file::unlink` and `fs::readdir` for possible error
/// conditions.
pub fn rmdir_recursive(path: &Path) {
    let children = readdir(path);
    for child in children.iter() {
        if child.is_dir() {
            rmdir_recursive(child);
        } else {
            unlink(child);
        }
    }
    // Directory should now be empty
    rmdir(path);
}

/// Changes the timestamps for a file's last modification and access time.
/// The file at the path specified will have its last access time set to
/// `atime` and its modification time set to `mtime`. The times specified should
/// be in milliseconds.
///
/// # Errors
///
/// This function will raise on the `io_error` condition if an error
/// happens.
// FIXME(#10301) these arguments should not be u64
pub fn change_file_times(path: &Path, atime: u64, mtime: u64) {
    LocalIo::maybe_raise(|io| io.fs_utime(&path.to_c_str(), atime, mtime));
}

impl Reader for File {
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
}

impl Writer for File {
    fn write(&mut self, buf: &[u8]) {
        match self.fd.write(buf) {
            Ok(()) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }
}

impl Seek for File {
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

    /// Whether the underlying implementation (be it a file path, or something
    /// else) points at a "regular file" on the FS. Will return false for paths
    /// to non-existent locations or directories or other non-regular files
    /// (named pipes, etc).
    ///
    /// # Errors
    ///
    /// Will not raise a condition
    pub fn is_file(&self) -> bool {
        match io::result(|| self.stat()) {
            Ok(s) => s.kind == io::TypeFile,
            Err(..) => false
        }
    }

    /// Whether the underlying implementation (be it a file path,
    /// or something else) is pointing at a directory in the underlying FS.
    /// Will return false for paths to non-existent locations or if the item is
    /// not a directory (eg files, named pipes, links, etc)
    ///
    /// # Errors
    ///
    /// Will not raise a condition
    pub fn is_dir(&self) -> bool {
        match io::result(|| self.stat()) {
            Ok(s) => s.kind == io::TypeDirectory,
            Err(..) => false
        }
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod test {
    use prelude::*;
    use io::{SeekSet, SeekCur, SeekEnd, io_error, Read, Open,
             ReadWrite};
    use io;
    use str;
    use io::fs::{File, rmdir, mkdir, readdir, rmdir_recursive,
                 mkdir_recursive, copy, unlink, stat, symlink, link,
                 readlink, chmod, lstat, change_file_times};
    use util;
    use path::Path;
    use io;
    use ops::Drop;

    struct TempDir(Path);

    impl TempDir {
        fn join(&self, path: &str) -> Path {
            let TempDir(ref p) = *self;
            p.join(path)
        }

        fn path<'a>(&'a self) -> &'a Path {
            let TempDir(ref p) = *self;
            p
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Gee, seeing how we're testing the fs module I sure hope that we
            // at least implement this correctly!
            let TempDir(ref p) = *self;
            io::fs::rmdir_recursive(p);
        }
    }

    pub fn tmpdir() -> TempDir {
        use os;
        use rand;
        let ret = os::tmpdir().join(format!("rust-{}", rand::random::<u32>()));
        io::fs::mkdir(&ret, io::UserRWX);
        TempDir(ret)
    }

    iotest!(fn file_test_io_smoke_test() {
        let message = "it's alright. have a good time";
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test.txt");
        {
            let mut write_stream = File::open_mode(filename, Open, ReadWrite);
            write_stream.write(message.as_bytes());
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            let mut read_buf = [0, .. 1028];
            let read_str = match read_stream.read(read_buf).unwrap() {
                -1|0 => fail!("shouldn't happen"),
                n => str::from_utf8_owned(read_buf.slice_to(n).to_owned())
            };
            assert_eq!(read_str, message.to_owned());
        }
        unlink(filename);
    })

    iotest!(fn invalid_path_raises() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_that_does_not_exist.txt");
        let mut called = false;
        io_error::cond.trap(|_| {
            called = true;
        }).inside(|| {
            let result = File::open_mode(filename, Open, Read);
            assert!(result.is_none());
        });
        assert!(called);
    })

    iotest!(fn file_test_iounlinking_invalid_path_should_raise_condition() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_another_file_that_does_not_exist.txt");
        let mut called = false;
        io_error::cond.trap(|_| {
            called = true;
        }).inside(|| unlink(filename));
        assert!(called);
    })

    iotest!(fn file_test_io_non_positional_read() {
        let message: &str = "ten-four";
        let mut read_mem = [0, .. 8];
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_positional.txt");
        {
            let mut rw_stream = File::open_mode(filename, Open, ReadWrite);
            rw_stream.write(message.as_bytes());
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
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
        assert_eq!(read_str, message);
    })

    iotest!(fn file_test_io_seek_and_tell_smoke_test() {
        let message = "ten-four";
        let mut read_mem = [0, .. 4];
        let set_cursor = 4 as u64;
        let mut tell_pos_pre_read;
        let mut tell_pos_post_read;
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_seeking.txt");
        {
            let mut rw_stream = File::open_mode(filename, Open, ReadWrite);
            rw_stream.write(message.as_bytes());
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            read_stream.seek(set_cursor as i64, SeekSet);
            tell_pos_pre_read = read_stream.tell();
            read_stream.read(read_mem);
            tell_pos_post_read = read_stream.tell();
        }
        unlink(filename);
        let read_str = str::from_utf8(read_mem);
        assert_eq!(read_str, message.slice(4, 8));
        assert_eq!(tell_pos_pre_read, set_cursor);
        assert_eq!(tell_pos_post_read, message.len() as u64);
    })

    iotest!(fn file_test_io_seek_and_write() {
        let initial_msg =   "food-is-yummy";
        let overwrite_msg =    "-the-bar!!";
        let final_msg =     "foo-the-bar!!";
        let seek_idx = 3;
        let mut read_mem = [0, .. 13];
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_seek_and_write.txt");
        {
            let mut rw_stream = File::open_mode(filename, Open, ReadWrite);
            rw_stream.write(initial_msg.as_bytes());
            rw_stream.seek(seek_idx as i64, SeekSet);
            rw_stream.write(overwrite_msg.as_bytes());
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            read_stream.read(read_mem);
        }
        unlink(filename);
        let read_str = str::from_utf8(read_mem);
        assert!(read_str == final_msg.to_owned());
    })

    iotest!(fn file_test_io_seek_shakedown() {
        use std::str;          // 01234567890123
        let initial_msg =   "qwer-asdf-zxcv";
        let chunk_one: &str = "qwer";
        let chunk_two: &str = "asdf";
        let chunk_three: &str = "zxcv";
        let mut read_mem = [0, .. 4];
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_seek_shakedown.txt");
        {
            let mut rw_stream = File::open_mode(filename, Open, ReadWrite);
            rw_stream.write(initial_msg.as_bytes());
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);

            read_stream.seek(-4, SeekEnd);
            read_stream.read(read_mem);
            assert_eq!(str::from_utf8(read_mem), chunk_three);

            read_stream.seek(-9, SeekCur);
            read_stream.read(read_mem);
            assert_eq!(str::from_utf8(read_mem), chunk_two);

            read_stream.seek(0, SeekSet);
            read_stream.read(read_mem);
            assert_eq!(str::from_utf8(read_mem), chunk_one);
        }
        unlink(filename);
    })

    iotest!(fn file_test_stat_is_correct_on_is_file() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_stat_correct_on_is_file.txt");
        {
            let mut fs = File::open_mode(filename, Open, ReadWrite);
            let msg = "hw";
            fs.write(msg.as_bytes());
        }
        let stat_res = stat(filename);
        assert_eq!(stat_res.kind, io::TypeFile);
        unlink(filename);
    })

    iotest!(fn file_test_stat_is_correct_on_is_dir() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_stat_correct_on_is_dir");
        mkdir(filename, io::UserRWX);
        let stat_res = filename.stat();
        assert!(stat_res.kind == io::TypeDirectory);
        rmdir(filename);
    })

    iotest!(fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("fileinfo_false_on_dir");
        mkdir(dir, io::UserRWX);
        assert!(dir.is_file() == false);
        rmdir(dir);
    })

    iotest!(fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
        let tmpdir = tmpdir();
        let file = &tmpdir.join("fileinfo_check_exists_b_and_a.txt");
        File::create(file).write(bytes!("foo"));
        assert!(file.exists());
        unlink(file);
        assert!(!file.exists());
    })

    iotest!(fn file_test_directoryinfo_check_exists_before_and_after_mkdir() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("before_and_after_dir");
        assert!(!dir.exists());
        mkdir(dir, io::UserRWX);
        assert!(dir.exists());
        assert!(dir.is_dir());
        rmdir(dir);
        assert!(!dir.exists());
    })

    iotest!(fn file_test_directoryinfo_readdir() {
        use std::str;
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("di_readdir");
        mkdir(dir, io::UserRWX);
        let prefix = "foo";
        for n in range(0,3) {
            let f = dir.join(format!("{}.txt", n));
            let mut w = File::create(&f);
            let msg_str = (prefix + n.to_str().to_owned()).to_owned();
            let msg = msg_str.as_bytes();
            w.write(msg);
        }
        let files = readdir(dir);
        let mut mem = [0u8, .. 4];
        for f in files.iter() {
            {
                let n = f.filestem_str();
                File::open(f).read(mem);
                let read_str = str::from_utf8(mem);
                let expected = match n {
                    None|Some("") => fail!("really shouldn't happen.."),
                    Some(n) => prefix+n
                };
                assert_eq!(expected.as_slice(), read_str);
            }
            unlink(f);
        }
        rmdir(dir);
    })

    iotest!(fn recursive_mkdir_slash() {
        mkdir_recursive(&Path::new("/"), io::UserRWX);
    })

    iotest!(fn unicode_path_is_dir() {
        assert!(Path::new(".").is_dir());
        assert!(!Path::new("test/stdtest/fs.rs").is_dir());

        let tmpdir = tmpdir();

        let mut dirpath = tmpdir.path().clone();
        dirpath.push(format!("test-가一ー你好"));
        mkdir(&dirpath, io::UserRWX);
        assert!(dirpath.is_dir());

        let mut filepath = dirpath;
        filepath.push("unicode-file-\uac00\u4e00\u30fc\u4f60\u597d.rs");
        File::create(&filepath); // ignore return; touch only
        assert!(!filepath.is_dir());
        assert!(filepath.exists());
    })

    iotest!(fn unicode_path_exists() {
        assert!(Path::new(".").exists());
        assert!(!Path::new("test/nonexistent-bogus-path").exists());

        let tmpdir = tmpdir();
        let unicode = tmpdir.path();
        let unicode = unicode.join(format!("test-각丁ー再见"));
        mkdir(&unicode, io::UserRWX);
        assert!(unicode.exists());
        assert!(!Path::new("test/unicode-bogus-path-각丁ー再见").exists());
    })

    iotest!(fn copy_file_does_not_exist() {
        let from = Path::new("test/nonexistent-bogus-path");
        let to = Path::new("test/other-bogus-path");
        match io::result(|| copy(&from, &to)) {
            Ok(..) => fail!(),
            Err(..) => {
                assert!(!from.exists());
                assert!(!to.exists());
            }
        }
    })

    iotest!(fn copy_file_ok() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        File::create(&input).write(bytes!("hello"));
        copy(&input, &out);
        let contents = File::open(&out).read_to_end();
        assert_eq!(contents.as_slice(), bytes!("hello"));

        assert_eq!(input.stat().perm, out.stat().perm);
    })

    iotest!(fn copy_file_dst_dir() {
        let tmpdir = tmpdir();
        let out = tmpdir.join("out");

        File::create(&out);
        match io::result(|| copy(&out, tmpdir.path())) {
            Ok(..) => fail!(), Err(..) => {}
        }
    })

    iotest!(fn copy_file_dst_exists() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in");
        let output = tmpdir.join("out");

        File::create(&input).write("foo".as_bytes());
        File::create(&output).write("bar".as_bytes());
        copy(&input, &output);

        assert_eq!(File::open(&output).read_to_end(),
                   (bytes!("foo")).to_owned());
    })

    iotest!(fn copy_file_src_dir() {
        let tmpdir = tmpdir();
        let out = tmpdir.join("out");

        match io::result(|| copy(tmpdir.path(), &out)) {
            Ok(..) => fail!(), Err(..) => {}
        }
        assert!(!out.exists());
    })

    iotest!(fn copy_file_preserves_perm_bits() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        File::create(&input);
        chmod(&input, io::UserRead);
        copy(&input, &out);
        assert!(out.stat().perm & io::UserWrite == 0);

        chmod(&input, io::UserFile);
        chmod(&out, io::UserFile);
    })

    #[cfg(not(windows))] // FIXME(#10264) operation not permitted?
    iotest!(fn symlinks_work() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        File::create(&input).write("foobar".as_bytes());
        symlink(&input, &out);
        if cfg!(not(windows)) {
            assert_eq!(lstat(&out).kind, io::TypeSymlink);
        }
        assert_eq!(stat(&out).size, stat(&input).size);
        assert_eq!(File::open(&out).read_to_end(), (bytes!("foobar")).to_owned());
    })

    #[cfg(not(windows))] // apparently windows doesn't like symlinks
    iotest!(fn symlink_noexist() {
        let tmpdir = tmpdir();
        // symlinks can point to things that don't exist
        symlink(&tmpdir.join("foo"), &tmpdir.join("bar"));
        assert!(readlink(&tmpdir.join("bar")).unwrap() == tmpdir.join("foo"));
    })

    iotest!(fn readlink_not_symlink() {
        let tmpdir = tmpdir();
        match io::result(|| readlink(tmpdir.path())) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }
    })

    iotest!(fn links_work() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        File::create(&input).write("foobar".as_bytes());
        link(&input, &out);
        if cfg!(not(windows)) {
            assert_eq!(lstat(&out).kind, io::TypeFile);
            assert_eq!(stat(&out).unstable.nlink, 2);
        }
        assert_eq!(stat(&out).size, stat(&input).size);
        assert_eq!(File::open(&out).read_to_end(), (bytes!("foobar")).to_owned());

        // can't link to yourself
        match io::result(|| link(&input, &input)) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }
        // can't link to something that doesn't exist
        match io::result(|| link(&tmpdir.join("foo"), &tmpdir.join("bar"))) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }
    })

    iotest!(fn chmod_works() {
        let tmpdir = tmpdir();
        let file = tmpdir.join("in.txt");

        File::create(&file);
        assert!(stat(&file).perm & io::UserWrite == io::UserWrite);
        chmod(&file, io::UserRead);
        assert!(stat(&file).perm & io::UserWrite == 0);

        match io::result(|| chmod(&tmpdir.join("foo"), io::UserRWX)) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }

        chmod(&file, io::UserFile);
    })

    iotest!(fn sync_doesnt_kill_anything() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let mut file = File::open_mode(&path, io::Open, io::ReadWrite).unwrap();
        file.fsync();
        file.datasync();
        file.write(bytes!("foo"));
        file.fsync();
        file.datasync();
        drop(file);
    })

    iotest!(fn truncate_works() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let mut file = File::open_mode(&path, io::Open, io::ReadWrite).unwrap();
        file.write(bytes!("foo"));
        file.fsync();

        // Do some simple things with truncation
        assert_eq!(stat(&path).size, 3);
        file.truncate(10);
        assert_eq!(stat(&path).size, 10);
        file.write(bytes!("bar"));
        file.fsync();
        assert_eq!(stat(&path).size, 10);
        assert_eq!(File::open(&path).read_to_end(),
                   (bytes!("foobar", 0, 0, 0, 0)).to_owned());

        // Truncate to a smaller length, don't seek, and then write something.
        // Ensure that the intermediate zeroes are all filled in (we're seeked
        // past the end of the file).
        file.truncate(2);
        assert_eq!(stat(&path).size, 2);
        file.write(bytes!("wut"));
        file.fsync();
        assert_eq!(stat(&path).size, 9);
        assert_eq!(File::open(&path).read_to_end(),
                   (bytes!("fo", 0, 0, 0, 0, "wut")).to_owned());
        drop(file);
    })

    iotest!(fn open_flavors() {
        let tmpdir = tmpdir();

        match io::result(|| File::open_mode(&tmpdir.join("a"), io::Open,
                                            io::Read)) {
            Ok(..) => fail!(), Err(..) => {}
        }
        File::open_mode(&tmpdir.join("b"), io::Open, io::Write).unwrap();
        File::open_mode(&tmpdir.join("c"), io::Open, io::ReadWrite).unwrap();
        File::open_mode(&tmpdir.join("d"), io::Append, io::Write).unwrap();
        File::open_mode(&tmpdir.join("e"), io::Append, io::ReadWrite).unwrap();
        File::open_mode(&tmpdir.join("f"), io::Truncate, io::Write).unwrap();
        File::open_mode(&tmpdir.join("g"), io::Truncate, io::ReadWrite).unwrap();

        File::create(&tmpdir.join("h")).write("foo".as_bytes());
        File::open_mode(&tmpdir.join("h"), io::Open, io::Read).unwrap();
        {
            let mut f = File::open_mode(&tmpdir.join("h"), io::Open,
                                        io::Read).unwrap();
            match io::result(|| f.write("wut".as_bytes())) {
                Ok(..) => fail!(), Err(..) => {}
            }
        }
        assert_eq!(stat(&tmpdir.join("h")).size, 3);
        {
            let mut f = File::open_mode(&tmpdir.join("h"), io::Append,
                                        io::Write).unwrap();
            f.write("bar".as_bytes());
        }
        assert_eq!(stat(&tmpdir.join("h")).size, 6);
        {
            let mut f = File::open_mode(&tmpdir.join("h"), io::Truncate,
                                        io::Write).unwrap();
            f.write("bar".as_bytes());
        }
        assert_eq!(stat(&tmpdir.join("h")).size, 3);
    })

    #[test]
    fn utime() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("a");
        File::create(&path);

        change_file_times(&path, 1000, 2000);
        assert_eq!(path.stat().accessed, 1000);
        assert_eq!(path.stat().modified, 2000);
    }

    #[test]
    fn utime_noexist() {
        let tmpdir = tmpdir();

        match io::result(|| change_file_times(&tmpdir.join("a"), 100, 200)) {
            Ok(..) => fail!(),
            Err(..) => {}
        }
    }
}
