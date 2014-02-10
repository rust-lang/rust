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
# #[allow(unused_must_use)];
use std::io::{File, fs};

let path = Path::new("foo.txt");

// create the file, whether it exists or not
let mut file = File::create(&path);
file.write(bytes!("foobar"));
# drop(file);

// open the file in read-only mode
let mut file = File::open(&path);
file.read_to_end();

println!("{}", path.stat().unwrap().size);
# drop(file);
fs::unlink(&path);
```

*/

use c_str::ToCStr;
use clone::Clone;
use iter::Iterator;
use super::{Reader, Writer, Seek};
use super::{SeekStyle, Read, Write, Open, IoError, Truncate,
            FileMode, FileAccess, FileStat, IoResult, FilePermission};
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
/// # Error
///
/// This type will return errors as an `IoResult<T>` if operations are
/// attempted against it for which its underlying file descriptor was not
/// configured at creation time, via the `FileAccess` parameter to
/// `File::open_mode()`.
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
    /// ```rust,should_fail
    /// use std::io::{File, Open, ReadWrite};
    ///
    /// let p = Path::new("/some/file/path.txt");
    ///
    /// let file = match File::open_mode(&p, Open, ReadWrite) {
    ///     Ok(f) => f,
    ///     Err(e) => fail!("file error: {}", e),
    /// };
    /// // do some stuff with that file
    ///
    /// // the file will be closed at the end of this block
    /// ```
    ///
    /// `FileMode` and `FileAccess` provide information about the permissions
    /// context in which a given stream is created. More information about them
    /// can be found in `std::io`'s docs. If a file is opened with `Write`
    /// or `ReadWrite` access, then it will be created it it does not already
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
                     access: FileAccess) -> IoResult<File> {
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
    /// ```rust
    /// use std::io::File;
    ///
    /// let contents = File::open(&Path::new("foo.txt")).read_to_end();
    /// ```
    pub fn open(path: &Path) -> IoResult<File> {
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
    /// ```rust
    /// # #[allow(unused_must_use)];
    /// use std::io::File;
    ///
    /// let mut f = File::create(&Path::new("foo.txt"));
    /// f.write(bytes!("This is a sample file"));
    /// # drop(f);
    /// # ::std::io::fs::unlink(&Path::new("foo.txt"));
    /// ```
    pub fn create(path: &Path) -> IoResult<File> {
        File::open_mode(path, Truncate, Write)
    }

    /// Returns the original path which was used to open this file.
    pub fn path<'a>(&'a self) -> &'a Path {
        &self.path
    }

    /// Synchronizes all modifications to this file to its permanent storage
    /// device. This will flush any internal buffers necessary to perform this
    /// operation.
    pub fn fsync(&mut self) -> IoResult<()> {
        self.fd.fsync()
    }

    /// This function is similar to `fsync`, except that it may not synchronize
    /// file metadata to the filesystem. This is intended for use case which
    /// must synchronize content, but don't need the metadata on disk. The goal
    /// of this method is to reduce disk operations.
    pub fn datasync(&mut self) -> IoResult<()> {
        self.fd.datasync()
    }

    /// Either truncates or extends the underlying file, updating the size of
    /// this file to become `size`. This is equivalent to unix's `truncate`
    /// function.
    ///
    /// If the `size` is less than the current file's size, then the file will
    /// be shrunk. If it is greater than the current file's size, then the file
    /// will be extended to `size` and have all of the intermediate data filled
    /// in with 0s.
    pub fn truncate(&mut self, size: i64) -> IoResult<()> {
        self.fd.truncate(size)
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
/// ```rust
/// # #[allow(unused_must_use)];
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
/// This function will return an error if the path points to a directory, the
/// user lacks permissions to remove the file, or if some other filesystem-level
/// error occurs.
pub fn unlink(path: &Path) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_unlink(&path.to_c_str()))
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
/// This call will return an error if the user lacks the requisite permissions
/// to perform a `stat` call on the given path or if there is no entry in the
/// filesystem at the provided path.
pub fn stat(path: &Path) -> IoResult<FileStat> {
    LocalIo::maybe_raise(|io| {
        io.fs_stat(&path.to_c_str())
    })
}

/// Perform the same operation as the `stat` function, except that this
/// function does not traverse through symlinks. This will return
/// information about the symlink file instead of the file that it points
/// to.
///
/// # Error
///
/// See `stat`
pub fn lstat(path: &Path) -> IoResult<FileStat> {
    LocalIo::maybe_raise(|io| {
        io.fs_lstat(&path.to_c_str())
    })
}

/// Rename a file or directory to a new name.
///
/// # Example
///
/// ```rust
/// # #[allow(unused_must_use)];
/// use std::io::fs;
///
/// fs::rename(&Path::new("foo"), &Path::new("bar"));
/// ```
///
/// # Error
///
/// Will return an error if the provided `path` doesn't exist, the process lacks
/// permissions to view the contents, or if some other intermittent I/O error
/// occurs.
pub fn rename(from: &Path, to: &Path) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_rename(&from.to_c_str(), &to.to_c_str()))
}

/// Copies the contents of one file to another. This function will also
/// copy the permission bits of the original file to the destination file.
///
/// Note that if `from` and `to` both point to the same file, then the file
/// will likely get truncated by this operation.
///
/// # Example
///
/// ```rust
/// # #[allow(unused_must_use)];
/// use std::io::fs;
///
/// fs::copy(&Path::new("foo.txt"), &Path::new("bar.txt"));
/// ```
///
/// # Error
///
/// Will return an error in the following situations, but is not limited to
/// just these cases:
///
/// * The `from` path is not a file
/// * The `from` file does not exist
/// * The current process does not have the permission rights to access
///   `from` or write `to`
///
/// Note that this copy is not atomic in that once the destination is
/// ensured to not exist, there is nothing preventing the destination from
/// being created and then destroyed by this operation.
pub fn copy(from: &Path, to: &Path) -> IoResult<()> {
    if !from.is_file() {
        return Err(IoError {
            kind: io::MismatchedFileTypeForOperation,
            desc: "the source path is not an existing file",
            detail: None,
        })
    }

    let mut reader = if_ok!(File::open(from));
    let mut writer = if_ok!(File::create(to));
    let mut buf = [0, ..io::DEFAULT_BUF_SIZE];

    loop {
        let amt = match reader.read(buf) {
            Ok(n) => n,
            Err(ref e) if e.kind == io::EndOfFile => { break }
            Err(e) => return Err(e)
        };
        if_ok!(writer.write(buf.slice_to(amt)));
    }

    chmod(to, if_ok!(from.stat()).perm)
}

/// Changes the permission mode bits found on a file or a directory. This
/// function takes a mask from the `io` module
///
/// # Example
///
/// ```rust
/// # #[allow(unused_must_use)];
/// use std::io;
/// use std::io::fs;
///
/// fs::chmod(&Path::new("file.txt"), io::UserFile);
/// fs::chmod(&Path::new("file.txt"), io::UserRead | io::UserWrite);
/// fs::chmod(&Path::new("dir"),      io::UserDir);
/// fs::chmod(&Path::new("file.exe"), io::UserExec);
/// ```
///
/// # Error
///
/// If this function encounters an I/O error, it will return an `Err` value.
/// Some possible error situations are not having the permission to
/// change the attributes of a file or the file not existing.
pub fn chmod(path: &Path, mode: io::FilePermission) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_chmod(&path.to_c_str(), mode))
}

/// Change the user and group owners of a file at the specified path.
pub fn chown(path: &Path, uid: int, gid: int) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_chown(&path.to_c_str(), uid, gid))
}

/// Creates a new hard link on the filesystem. The `dst` path will be a
/// link pointing to the `src` path. Note that systems often require these
/// two paths to both be located on the same filesystem.
pub fn link(src: &Path, dst: &Path) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_link(&src.to_c_str(), &dst.to_c_str()))
}

/// Creates a new symbolic link on the filesystem. The `dst` path will be a
/// symlink pointing to the `src` path.
pub fn symlink(src: &Path, dst: &Path) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_symlink(&src.to_c_str(), &dst.to_c_str()))
}

/// Reads a symlink, returning the file that the symlink points to.
///
/// # Error
///
/// This function will return an error on failure. Failure conditions include
/// reading a file that does not exist or reading a file which is not a symlink.
pub fn readlink(path: &Path) -> IoResult<Path> {
    LocalIo::maybe_raise(|io| io.fs_readlink(&path.to_c_str()))
}

/// Create a new, empty directory at the provided path
///
/// # Example
///
/// ```rust
/// # #[allow(unused_must_use)];
/// use std::io;
/// use std::io::fs;
///
/// let p = Path::new("/some/dir");
/// fs::mkdir(&p, io::UserRWX);
/// ```
///
/// # Error
///
/// This call will return an error if the user lacks permissions to make a new
/// directory at the provided path, or if the directory already exists.
pub fn mkdir(path: &Path, mode: FilePermission) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_mkdir(&path.to_c_str(), mode))
}

/// Remove an existing, empty directory
///
/// # Example
///
/// ```rust
/// # #[allow(unused_must_use)];
/// use std::io::fs;
///
/// let p = Path::new("/some/dir");
/// fs::rmdir(&p);
/// ```
///
/// # Error
///
/// This call will return an error if the user lacks permissions to remove the
/// directory at the provided path, or if the directory isn't empty.
pub fn rmdir(path: &Path) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_rmdir(&path.to_c_str()))
}

/// Retrieve a vector containing all entries within a provided directory
///
/// # Example
///
/// ```rust
/// use std::io;
/// use std::io::fs;
///
/// // one possible implementation of fs::walk_dir only visiting files
/// fn visit_dirs(dir: &Path, cb: |&Path|) -> io::IoResult<()> {
///     if dir.is_dir() {
///         let contents = if_ok!(fs::readdir(dir));
///         for entry in contents.iter() {
///             if entry.is_dir() {
///                 if_ok!(visit_dirs(entry, |p| cb(p)));
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
/// Will return an error if the provided `from` doesn't exist, the process lacks
/// permissions to view the contents or if the `path` points at a non-directory
/// file
pub fn readdir(path: &Path) -> IoResult<~[Path]> {
    LocalIo::maybe_raise(|io| {
        io.fs_readdir(&path.to_c_str(), 0)
    })
}

/// Returns an iterator which will recursively walk the directory structure
/// rooted at `path`. The path given will not be iterated over, and this will
/// perform iteration in a top-down order.
pub fn walk_dir(path: &Path) -> IoResult<Directories> {
    Ok(Directories { stack: if_ok!(readdir(path)) })
}

/// An iterator which walks over a directory
pub struct Directories {
    priv stack: ~[Path],
}

impl Iterator<Path> for Directories {
    fn next(&mut self) -> Option<Path> {
        match self.stack.shift() {
            Some(path) => {
                if path.is_dir() {
                    match readdir(&path) {
                        Ok(dirs) => { self.stack.push_all_move(dirs); }
                        Err(..) => {}
                    }
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
/// # Error
///
/// This function will return an `Err` value if an error happens, see
/// `fs::mkdir` for more information about error conditions and performance.
pub fn mkdir_recursive(path: &Path, mode: FilePermission) -> IoResult<()> {
    // tjc: if directory exists but with different permissions,
    // should we return false?
    if path.is_dir() {
        return Ok(())
    }
    if path.filename().is_some() {
        if_ok!(mkdir_recursive(&path.dir_path(), mode));
    }
    mkdir(path, mode)
}

/// Removes a directory at this path, after removing all its contents. Use
/// carefully!
///
/// # Error
///
/// This function will return an `Err` value if an error happens. See
/// `file::unlink` and `fs::readdir` for possible error conditions.
pub fn rmdir_recursive(path: &Path) -> IoResult<()> {
    let children = if_ok!(readdir(path));
    for child in children.iter() {
        if child.is_dir() {
            if_ok!(rmdir_recursive(child));
        } else {
            if_ok!(unlink(child));
        }
    }
    // Directory should now be empty
    rmdir(path)
}

/// Changes the timestamps for a file's last modification and access time.
/// The file at the path specified will have its last access time set to
/// `atime` and its modification time set to `mtime`. The times specified should
/// be in milliseconds.
// FIXME(#10301) these arguments should not be u64
pub fn change_file_times(path: &Path, atime: u64, mtime: u64) -> IoResult<()> {
    LocalIo::maybe_raise(|io| io.fs_utime(&path.to_c_str(), atime, mtime))
}

impl Reader for File {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        match self.fd.read(buf) {
            Ok(read) => {
                self.last_nread = read;
                match read {
                    0 => Err(io::standard_error(io::EndOfFile)),
                    _ => Ok(read as uint)
                }
            },
            Err(e) => Err(e),
        }
    }
}

impl Writer for File {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.fd.write(buf) }
}

impl Seek for File {
    fn tell(&self) -> IoResult<u64> {
        self.fd.tell()
    }

    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> {
        match self.fd.seek(pos, style) {
            Ok(_) => {
                // successful seek resets EOF indicator
                self.last_nread = -1;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

impl path::Path {
    /// Get information on the file, directory, etc at this path.
    ///
    /// Consult the `file::stat` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with `file::stat`.
    pub fn stat(&self) -> IoResult<FileStat> { stat(self) }

    /// Boolean value indicator whether the underlying file exists on the local
    /// filesystem. This will return true if the path points to either a
    /// directory or a file.
    ///
    /// # Error
    ///
    /// Will not raise a condition
    pub fn exists(&self) -> bool {
        self.stat().is_ok()
    }

    /// Whether the underlying implementation (be it a file path, or something
    /// else) points at a "regular file" on the FS. Will return false for paths
    /// to non-existent locations or directories or other non-regular files
    /// (named pipes, etc).
    ///
    /// # Error
    ///
    /// Will not raise a condition
    pub fn is_file(&self) -> bool {
        match self.stat() {
            Ok(s) => s.kind == io::TypeFile,
            Err(..) => false
        }
    }

    /// Whether the underlying implementation (be it a file path,
    /// or something else) is pointing at a directory in the underlying FS.
    /// Will return false for paths to non-existent locations or if the item is
    /// not a directory (eg files, named pipes, links, etc)
    ///
    /// # Error
    ///
    /// Will not raise a condition
    pub fn is_dir(&self) -> bool {
        match self.stat() {
            Ok(s) => s.kind == io::TypeDirectory,
            Err(..) => false
        }
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod test {
    use prelude::*;
    use io::{SeekSet, SeekCur, SeekEnd, Read, Open, ReadWrite};
    use io;
    use str;
    use io::fs::{File, rmdir, mkdir, readdir, rmdir_recursive,
                 mkdir_recursive, copy, unlink, stat, symlink, link,
                 readlink, chmod, lstat, change_file_times};
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
            io::fs::rmdir_recursive(p).unwrap();
        }
    }

    pub fn tmpdir() -> TempDir {
        use os;
        use rand;
        let ret = os::tmpdir().join(format!("rust-{}", rand::random::<u32>()));
        io::fs::mkdir(&ret, io::UserRWX).unwrap();
        TempDir(ret)
    }

    iotest!(fn file_test_io_smoke_test() {
        let message = "it's alright. have a good time";
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test.txt");
        {
            let mut write_stream = File::open_mode(filename, Open, ReadWrite);
            write_stream.write(message.as_bytes()).unwrap();
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            let mut read_buf = [0, .. 1028];
            let read_str = match read_stream.read(read_buf).unwrap() {
                -1|0 => fail!("shouldn't happen"),
                n => str::from_utf8_owned(read_buf.slice_to(n).to_owned()).unwrap()
            };
            assert_eq!(read_str, message.to_owned());
        }
        unlink(filename).unwrap();
    })

    iotest!(fn invalid_path_raises() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_that_does_not_exist.txt");
        let result = File::open_mode(filename, Open, Read);
        assert!(result.is_err());
    })

    iotest!(fn file_test_iounlinking_invalid_path_should_raise_condition() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_another_file_that_does_not_exist.txt");
        assert!(unlink(filename).is_err());
    })

    iotest!(fn file_test_io_non_positional_read() {
        let message: &str = "ten-four";
        let mut read_mem = [0, .. 8];
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test_positional.txt");
        {
            let mut rw_stream = File::open_mode(filename, Open, ReadWrite);
            rw_stream.write(message.as_bytes()).unwrap();
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            {
                let read_buf = read_mem.mut_slice(0, 4);
                read_stream.read(read_buf).unwrap();
            }
            {
                let read_buf = read_mem.mut_slice(4, 8);
                read_stream.read(read_buf).unwrap();
            }
        }
        unlink(filename).unwrap();
        let read_str = str::from_utf8(read_mem).unwrap();
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
            rw_stream.write(message.as_bytes()).unwrap();
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            read_stream.seek(set_cursor as i64, SeekSet).unwrap();
            tell_pos_pre_read = read_stream.tell().unwrap();
            read_stream.read(read_mem).unwrap();
            tell_pos_post_read = read_stream.tell().unwrap();
        }
        unlink(filename).unwrap();
        let read_str = str::from_utf8(read_mem).unwrap();
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
            rw_stream.write(initial_msg.as_bytes()).unwrap();
            rw_stream.seek(seek_idx as i64, SeekSet).unwrap();
            rw_stream.write(overwrite_msg.as_bytes()).unwrap();
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            read_stream.read(read_mem).unwrap();
        }
        unlink(filename).unwrap();
        let read_str = str::from_utf8(read_mem).unwrap();
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
            rw_stream.write(initial_msg.as_bytes()).unwrap();
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);

            read_stream.seek(-4, SeekEnd).unwrap();
            read_stream.read(read_mem).unwrap();
            assert_eq!(str::from_utf8(read_mem).unwrap(), chunk_three);

            read_stream.seek(-9, SeekCur).unwrap();
            read_stream.read(read_mem).unwrap();
            assert_eq!(str::from_utf8(read_mem).unwrap(), chunk_two);

            read_stream.seek(0, SeekSet).unwrap();
            read_stream.read(read_mem).unwrap();
            assert_eq!(str::from_utf8(read_mem).unwrap(), chunk_one);
        }
        unlink(filename).unwrap();
    })

    iotest!(fn file_test_stat_is_correct_on_is_file() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_stat_correct_on_is_file.txt");
        {
            let mut fs = File::open_mode(filename, Open, ReadWrite);
            let msg = "hw";
            fs.write(msg.as_bytes()).unwrap();
        }
        let stat_res = stat(filename).unwrap();
        assert_eq!(stat_res.kind, io::TypeFile);
        unlink(filename).unwrap();
    })

    iotest!(fn file_test_stat_is_correct_on_is_dir() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_stat_correct_on_is_dir");
        mkdir(filename, io::UserRWX).unwrap();
        let stat_res = filename.stat().unwrap();
        assert!(stat_res.kind == io::TypeDirectory);
        rmdir(filename).unwrap();
    })

    iotest!(fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("fileinfo_false_on_dir");
        mkdir(dir, io::UserRWX).unwrap();
        assert!(dir.is_file() == false);
        rmdir(dir).unwrap();
    })

    iotest!(fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
        let tmpdir = tmpdir();
        let file = &tmpdir.join("fileinfo_check_exists_b_and_a.txt");
        File::create(file).write(bytes!("foo")).unwrap();
        assert!(file.exists());
        unlink(file).unwrap();
        assert!(!file.exists());
    })

    iotest!(fn file_test_directoryinfo_check_exists_before_and_after_mkdir() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("before_and_after_dir");
        assert!(!dir.exists());
        mkdir(dir, io::UserRWX).unwrap();
        assert!(dir.exists());
        assert!(dir.is_dir());
        rmdir(dir).unwrap();
        assert!(!dir.exists());
    })

    iotest!(fn file_test_directoryinfo_readdir() {
        use std::str;
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("di_readdir");
        mkdir(dir, io::UserRWX).unwrap();
        let prefix = "foo";
        for n in range(0,3) {
            let f = dir.join(format!("{}.txt", n));
            let mut w = File::create(&f).unwrap();
            let msg_str = (prefix + n.to_str().to_owned()).to_owned();
            let msg = msg_str.as_bytes();
            w.write(msg).unwrap();
        }
        let files = readdir(dir).unwrap();
        let mut mem = [0u8, .. 4];
        for f in files.iter() {
            {
                let n = f.filestem_str();
                File::open(f).read(mem).unwrap();
                let read_str = str::from_utf8(mem).unwrap();
                let expected = match n {
                    None|Some("") => fail!("really shouldn't happen.."),
                    Some(n) => prefix+n
                };
                assert_eq!(expected.as_slice(), read_str);
            }
            unlink(f).unwrap();
        }
        rmdir(dir).unwrap();
    })

    iotest!(fn recursive_mkdir_slash() {
        mkdir_recursive(&Path::new("/"), io::UserRWX).unwrap();
    })

    iotest!(fn unicode_path_is_dir() {
        assert!(Path::new(".").is_dir());
        assert!(!Path::new("test/stdtest/fs.rs").is_dir());

        let tmpdir = tmpdir();

        let mut dirpath = tmpdir.path().clone();
        dirpath.push(format!("test-가一ー你好"));
        mkdir(&dirpath, io::UserRWX).unwrap();
        assert!(dirpath.is_dir());

        let mut filepath = dirpath;
        filepath.push("unicode-file-\uac00\u4e00\u30fc\u4f60\u597d.rs");
        File::create(&filepath).unwrap(); // ignore return; touch only
        assert!(!filepath.is_dir());
        assert!(filepath.exists());
    })

    iotest!(fn unicode_path_exists() {
        assert!(Path::new(".").exists());
        assert!(!Path::new("test/nonexistent-bogus-path").exists());

        let tmpdir = tmpdir();
        let unicode = tmpdir.path();
        let unicode = unicode.join(format!("test-각丁ー再见"));
        mkdir(&unicode, io::UserRWX).unwrap();
        assert!(unicode.exists());
        assert!(!Path::new("test/unicode-bogus-path-각丁ー再见").exists());
    })

    iotest!(fn copy_file_does_not_exist() {
        let from = Path::new("test/nonexistent-bogus-path");
        let to = Path::new("test/other-bogus-path");
        match copy(&from, &to) {
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

        File::create(&input).write(bytes!("hello")).unwrap();
        copy(&input, &out).unwrap();
        let contents = File::open(&out).read_to_end().unwrap();
        assert_eq!(contents.as_slice(), bytes!("hello"));

        assert_eq!(input.stat().unwrap().perm, out.stat().unwrap().perm);
    })

    iotest!(fn copy_file_dst_dir() {
        let tmpdir = tmpdir();
        let out = tmpdir.join("out");

        File::create(&out).unwrap();
        match copy(&out, tmpdir.path()) {
            Ok(..) => fail!(), Err(..) => {}
        }
    })

    iotest!(fn copy_file_dst_exists() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in");
        let output = tmpdir.join("out");

        File::create(&input).write("foo".as_bytes()).unwrap();
        File::create(&output).write("bar".as_bytes()).unwrap();
        copy(&input, &output).unwrap();

        assert_eq!(File::open(&output).read_to_end().unwrap(),
                   (bytes!("foo")).to_owned());
    })

    iotest!(fn copy_file_src_dir() {
        let tmpdir = tmpdir();
        let out = tmpdir.join("out");

        match copy(tmpdir.path(), &out) {
            Ok(..) => fail!(), Err(..) => {}
        }
        assert!(!out.exists());
    })

    iotest!(fn copy_file_preserves_perm_bits() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        File::create(&input).unwrap();
        chmod(&input, io::UserRead).unwrap();
        copy(&input, &out).unwrap();
        assert!(out.stat().unwrap().perm & io::UserWrite == 0);

        chmod(&input, io::UserFile).unwrap();
        chmod(&out, io::UserFile).unwrap();
    })

    #[cfg(not(windows))] // FIXME(#10264) operation not permitted?
    iotest!(fn symlinks_work() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        File::create(&input).write("foobar".as_bytes()).unwrap();
        symlink(&input, &out).unwrap();
        if cfg!(not(windows)) {
            assert_eq!(lstat(&out).unwrap().kind, io::TypeSymlink);
        }
        assert_eq!(stat(&out).unwrap().size, stat(&input).unwrap().size);
        assert_eq!(File::open(&out).read_to_end().unwrap(),
                   (bytes!("foobar")).to_owned());
    })

    #[cfg(not(windows))] // apparently windows doesn't like symlinks
    iotest!(fn symlink_noexist() {
        let tmpdir = tmpdir();
        // symlinks can point to things that don't exist
        symlink(&tmpdir.join("foo"), &tmpdir.join("bar")).unwrap();
        assert!(readlink(&tmpdir.join("bar")).unwrap() == tmpdir.join("foo"));
    })

    iotest!(fn readlink_not_symlink() {
        let tmpdir = tmpdir();
        match readlink(tmpdir.path()) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }
    })

    iotest!(fn links_work() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        File::create(&input).write("foobar".as_bytes()).unwrap();
        link(&input, &out).unwrap();
        if cfg!(not(windows)) {
            assert_eq!(lstat(&out).unwrap().kind, io::TypeFile);
            assert_eq!(stat(&out).unwrap().unstable.nlink, 2);
        }
        assert_eq!(stat(&out).unwrap().size, stat(&input).unwrap().size);
        assert_eq!(File::open(&out).read_to_end().unwrap(),
                   (bytes!("foobar")).to_owned());

        // can't link to yourself
        match link(&input, &input) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }
        // can't link to something that doesn't exist
        match link(&tmpdir.join("foo"), &tmpdir.join("bar")) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }
    })

    iotest!(fn chmod_works() {
        let tmpdir = tmpdir();
        let file = tmpdir.join("in.txt");

        File::create(&file).unwrap();
        assert!(stat(&file).unwrap().perm & io::UserWrite == io::UserWrite);
        chmod(&file, io::UserRead).unwrap();
        assert!(stat(&file).unwrap().perm & io::UserWrite == 0);

        match chmod(&tmpdir.join("foo"), io::UserRWX) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }

        chmod(&file, io::UserFile).unwrap();
    })

    iotest!(fn sync_doesnt_kill_anything() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let mut file = File::open_mode(&path, io::Open, io::ReadWrite).unwrap();
        file.fsync().unwrap();
        file.datasync().unwrap();
        file.write(bytes!("foo")).unwrap();
        file.fsync().unwrap();
        file.datasync().unwrap();
        drop(file);
    })

    iotest!(fn truncate_works() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let mut file = File::open_mode(&path, io::Open, io::ReadWrite).unwrap();
        file.write(bytes!("foo")).unwrap();
        file.fsync().unwrap();

        // Do some simple things with truncation
        assert_eq!(stat(&path).unwrap().size, 3);
        file.truncate(10).unwrap();
        assert_eq!(stat(&path).unwrap().size, 10);
        file.write(bytes!("bar")).unwrap();
        file.fsync().unwrap();
        assert_eq!(stat(&path).unwrap().size, 10);
        assert_eq!(File::open(&path).read_to_end().unwrap(),
                   (bytes!("foobar", 0, 0, 0, 0)).to_owned());

        // Truncate to a smaller length, don't seek, and then write something.
        // Ensure that the intermediate zeroes are all filled in (we're seeked
        // past the end of the file).
        file.truncate(2).unwrap();
        assert_eq!(stat(&path).unwrap().size, 2);
        file.write(bytes!("wut")).unwrap();
        file.fsync().unwrap();
        assert_eq!(stat(&path).unwrap().size, 9);
        assert_eq!(File::open(&path).read_to_end().unwrap(),
                   (bytes!("fo", 0, 0, 0, 0, "wut")).to_owned());
        drop(file);
    })

    iotest!(fn open_flavors() {
        let tmpdir = tmpdir();

        match File::open_mode(&tmpdir.join("a"), io::Open, io::Read) {
            Ok(..) => fail!(), Err(..) => {}
        }
        File::open_mode(&tmpdir.join("b"), io::Open, io::Write).unwrap();
        File::open_mode(&tmpdir.join("c"), io::Open, io::ReadWrite).unwrap();
        File::open_mode(&tmpdir.join("d"), io::Append, io::Write).unwrap();
        File::open_mode(&tmpdir.join("e"), io::Append, io::ReadWrite).unwrap();
        File::open_mode(&tmpdir.join("f"), io::Truncate, io::Write).unwrap();
        File::open_mode(&tmpdir.join("g"), io::Truncate, io::ReadWrite).unwrap();

        File::create(&tmpdir.join("h")).write("foo".as_bytes()).unwrap();
        File::open_mode(&tmpdir.join("h"), io::Open, io::Read).unwrap();
        {
            let mut f = File::open_mode(&tmpdir.join("h"), io::Open,
                                        io::Read).unwrap();
            match f.write("wut".as_bytes()) {
                Ok(..) => fail!(), Err(..) => {}
            }
        }
        assert_eq!(stat(&tmpdir.join("h")).unwrap().size, 3);
        {
            let mut f = File::open_mode(&tmpdir.join("h"), io::Append,
                                        io::Write).unwrap();
            f.write("bar".as_bytes()).unwrap();
        }
        assert_eq!(stat(&tmpdir.join("h")).unwrap().size, 6);
        {
            let mut f = File::open_mode(&tmpdir.join("h"), io::Truncate,
                                        io::Write).unwrap();
            f.write("bar".as_bytes()).unwrap();
        }
        assert_eq!(stat(&tmpdir.join("h")).unwrap().size, 3);
    })

    #[test]
    fn utime() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("a");
        File::create(&path).unwrap();

        change_file_times(&path, 1000, 2000).unwrap();
        assert_eq!(path.stat().unwrap().accessed, 1000);
        assert_eq!(path.stat().unwrap().modified, 2000);
    }

    #[test]
    fn utime_noexist() {
        let tmpdir = tmpdir();

        match change_file_times(&tmpdir.join("a"), 100, 200) {
            Ok(..) => fail!(),
            Err(..) => {}
        }
    }
}
