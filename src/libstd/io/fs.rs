// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
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
# #![allow(unused_must_use)]
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
use container::Container;
use iter::Iterator;
use kinds::Send;
use super::{Reader, Writer, Seek};
use super::{SeekStyle, Read, Write, Open, IoError, Truncate};
use super::{FileMode, FileAccess, FileStat, IoResult, FilePermission};
use rt::rtio::{RtioFileStream, IoFactory, LocalIo};
use io;
use option::{Some, None, Option};
use owned::Box;
use result::{Ok, Err};
use path;
use path::{Path, GenericPath};
use slice::{OwnedVector, ImmutableVector};
use vec::Vec;

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
    fd: Box<RtioFileStream:Send>,
    path: Path,
    last_nread: int,
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
    /// or `ReadWrite` access, then it will be created it does not already
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
    /// # #![allow(unused_must_use)]
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

    /// Queries information about the underlying file.
    pub fn stat(&mut self) -> IoResult<FileStat> {
        self.fd.fstat()
    }
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
/// # #![allow(unused_must_use)]
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
/// # #![allow(unused_must_use)]
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

    let mut reader = try!(File::open(from));
    let mut writer = try!(File::create(to));
    let mut buf = [0, ..io::DEFAULT_BUF_SIZE];

    loop {
        let amt = match reader.read(buf) {
            Ok(n) => n,
            Err(ref e) if e.kind == io::EndOfFile => { break }
            Err(e) => return Err(e)
        };
        try!(writer.write(buf.slice_to(amt)));
    }

    chmod(to, try!(from.stat()).perm)
}

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
/// # #![allow(unused_must_use)]
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
/// # #![allow(unused_must_use)]
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
/// Will return an error if the provided `from` doesn't exist, the process lacks
/// permissions to view the contents or if the `path` points at a non-directory
/// file
pub fn readdir(path: &Path) -> IoResult<Vec<Path>> {
    LocalIo::maybe_raise(|io| {
        io.fs_readdir(&path.to_c_str(), 0)
    })
}

/// Returns an iterator which will recursively walk the directory structure
/// rooted at `path`. The path given will not be iterated over, and this will
/// perform iteration in some top-down order.  The contents of unreadable
/// subdirectories are ignored.
pub fn walk_dir(path: &Path) -> IoResult<Directories> {
    Ok(Directories { stack: try!(readdir(path)) })
}

/// An iterator which walks over a directory
pub struct Directories {
    stack: Vec<Path>,
}

impl Iterator<Path> for Directories {
    fn next(&mut self) -> Option<Path> {
        match self.stack.pop() {
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

    let mut comps = path.components();
    let mut curpath = path.root_path().unwrap_or(Path::new("."));

    for c in comps {
        curpath.push(c);

        match mkdir(&curpath, mode) {
            Err(mkdir_err) => {
                // already exists ?
                if try!(stat(&curpath)).kind != io::TypeDirectory {
                    return Err(mkdir_err);
                }
            }
            Ok(()) => ()
        }
    }

    Ok(())
}

/// Removes a directory at this path, after removing all its contents. Use
/// carefully!
///
/// # Error
///
/// This function will return an `Err` value if an error happens. See
/// `file::unlink` and `fs::readdir` for possible error conditions.
pub fn rmdir_recursive(path: &Path) -> IoResult<()> {
    let mut rm_stack = Vec::new();
    rm_stack.push(path.clone());

    while !rm_stack.is_empty() {
        let children = try!(readdir(rm_stack.last().unwrap()));
        let mut has_child_dir = false;

        // delete all regular files in the way and push subdirs
        // on the stack
        for child in children.move_iter() {
            // FIXME(#12795) we should use lstat in all cases
            let child_type = match cfg!(windows) {
                true => try!(stat(&child)).kind,
                false => try!(lstat(&child)).kind
            };

            if child_type == io::TypeDirectory {
                rm_stack.push(child);
                has_child_dir = true;
            } else {
                // we can carry on safely if the file is already gone
                // (eg: deleted by someone else since readdir)
                match unlink(&child) {
                    Ok(()) => (),
                    Err(ref e) if e.kind == io::FileNotFound => (),
                    Err(e) => return Err(e)
                }
            }
        }

        // if no subdir was found, let's pop and delete
        if !has_child_dir {
            match rmdir(&rm_stack.pop().unwrap()) {
                Ok(()) => (),
                Err(ref e) if e.kind == io::FileNotFound => (),
                Err(e) => return Err(e)
            }
        }
    }

    Ok(())
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
    /// Consult the `fs::stat` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with `file::stat`.
    pub fn stat(&self) -> IoResult<FileStat> { stat(self) }

    /// Get information on the file, directory, etc at this path, not following
    /// symlinks.
    ///
    /// Consult the `fs::lstat` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with `file::lstat`.
    pub fn lstat(&self) -> IoResult<FileStat> { lstat(self) }

    /// Boolean value indicator whether the underlying file exists on the local
    /// filesystem. Returns false in exactly the cases where `fs::stat` fails.
    pub fn exists(&self) -> bool {
        self.stat().is_ok()
    }

    /// Whether the underlying implementation (be it a file path, or something
    /// else) points at a "regular file" on the FS. Will return false for paths
    /// to non-existent locations or directories or other non-regular files
    /// (named pipes, etc). Follows links when making this determination.
    pub fn is_file(&self) -> bool {
        match self.stat() {
            Ok(s) => s.kind == io::TypeFile,
            Err(..) => false
        }
    }

    /// Whether the underlying implementation (be it a file path, or something
    /// else) is pointing at a directory in the underlying FS. Will return
    /// false for paths to non-existent locations or if the item is not a
    /// directory (eg files, named pipes, etc). Follows links when making this
    /// determination.
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
    use str::StrSlice;

    macro_rules! check( ($e:expr) => (
        match $e {
            Ok(t) => t,
            Err(e) => fail!("{} failed with: {}", stringify!($e), e),
        }
    ) )

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
            check!(io::fs::rmdir_recursive(p));
        }
    }

    pub fn tmpdir() -> TempDir {
        use os;
        use rand;
        let ret = os::tmpdir().join(format!("rust-{}", rand::random::<u32>()));
        check!(io::fs::mkdir(&ret, io::UserRWX));
        TempDir(ret)
    }

    iotest!(fn file_test_io_smoke_test() {
        let message = "it's alright. have a good time";
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_rt_io_file_test.txt");
        {
            let mut write_stream = File::open_mode(filename, Open, ReadWrite);
            check!(write_stream.write(message.as_bytes()));
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            let mut read_buf = [0, .. 1028];
            let read_str = match check!(read_stream.read(read_buf)) {
                -1|0 => fail!("shouldn't happen"),
                n => str::from_utf8(read_buf.slice_to(n).to_owned()).unwrap().to_owned()
            };
            assert_eq!(read_str, message.to_owned());
        }
        check!(unlink(filename));
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
            check!(rw_stream.write(message.as_bytes()));
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            {
                let read_buf = read_mem.mut_slice(0, 4);
                check!(read_stream.read(read_buf));
            }
            {
                let read_buf = read_mem.mut_slice(4, 8);
                check!(read_stream.read(read_buf));
            }
        }
        check!(unlink(filename));
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
            check!(rw_stream.write(message.as_bytes()));
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            check!(read_stream.seek(set_cursor as i64, SeekSet));
            tell_pos_pre_read = check!(read_stream.tell());
            check!(read_stream.read(read_mem));
            tell_pos_post_read = check!(read_stream.tell());
        }
        check!(unlink(filename));
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
            check!(rw_stream.write(initial_msg.as_bytes()));
            check!(rw_stream.seek(seek_idx as i64, SeekSet));
            check!(rw_stream.write(overwrite_msg.as_bytes()));
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);
            check!(read_stream.read(read_mem));
        }
        check!(unlink(filename));
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
            check!(rw_stream.write(initial_msg.as_bytes()));
        }
        {
            let mut read_stream = File::open_mode(filename, Open, Read);

            check!(read_stream.seek(-4, SeekEnd));
            check!(read_stream.read(read_mem));
            assert_eq!(str::from_utf8(read_mem).unwrap(), chunk_three);

            check!(read_stream.seek(-9, SeekCur));
            check!(read_stream.read(read_mem));
            assert_eq!(str::from_utf8(read_mem).unwrap(), chunk_two);

            check!(read_stream.seek(0, SeekSet));
            check!(read_stream.read(read_mem));
            assert_eq!(str::from_utf8(read_mem).unwrap(), chunk_one);
        }
        check!(unlink(filename));
    })

    iotest!(fn file_test_stat_is_correct_on_is_file() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_stat_correct_on_is_file.txt");
        {
            let mut fs = check!(File::open_mode(filename, Open, ReadWrite));
            let msg = "hw";
            fs.write(msg.as_bytes()).unwrap();

            let fstat_res = check!(fs.stat());
            assert_eq!(fstat_res.kind, io::TypeFile);
        }
        let stat_res_fn = check!(stat(filename));
        assert_eq!(stat_res_fn.kind, io::TypeFile);
        let stat_res_meth = check!(filename.stat());
        assert_eq!(stat_res_meth.kind, io::TypeFile);
        check!(unlink(filename));
    })

    iotest!(fn file_test_stat_is_correct_on_is_dir() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_stat_correct_on_is_dir");
        check!(mkdir(filename, io::UserRWX));
        let stat_res_fn = check!(stat(filename));
        assert!(stat_res_fn.kind == io::TypeDirectory);
        let stat_res_meth = check!(filename.stat());
        assert!(stat_res_meth.kind == io::TypeDirectory);
        check!(rmdir(filename));
    })

    iotest!(fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("fileinfo_false_on_dir");
        check!(mkdir(dir, io::UserRWX));
        assert!(dir.is_file() == false);
        check!(rmdir(dir));
    })

    iotest!(fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
        let tmpdir = tmpdir();
        let file = &tmpdir.join("fileinfo_check_exists_b_and_a.txt");
        check!(File::create(file).write(bytes!("foo")));
        assert!(file.exists());
        check!(unlink(file));
        assert!(!file.exists());
    })

    iotest!(fn file_test_directoryinfo_check_exists_before_and_after_mkdir() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("before_and_after_dir");
        assert!(!dir.exists());
        check!(mkdir(dir, io::UserRWX));
        assert!(dir.exists());
        assert!(dir.is_dir());
        check!(rmdir(dir));
        assert!(!dir.exists());
    })

    iotest!(fn file_test_directoryinfo_readdir() {
        use std::str;
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("di_readdir");
        check!(mkdir(dir, io::UserRWX));
        let prefix = "foo";
        for n in range(0,3) {
            let f = dir.join(format!("{}.txt", n));
            let mut w = check!(File::create(&f));
            let msg_str = (prefix + n.to_str().to_owned()).to_owned();
            let msg = msg_str.as_bytes();
            check!(w.write(msg));
        }
        let files = check!(readdir(dir));
        let mut mem = [0u8, .. 4];
        for f in files.iter() {
            {
                let n = f.filestem_str();
                check!(File::open(f).read(mem));
                let read_str = str::from_utf8(mem).unwrap();
                let expected = match n {
                    None|Some("") => fail!("really shouldn't happen.."),
                    Some(n) => prefix+n
                };
                assert_eq!(expected.as_slice(), read_str);
            }
            check!(unlink(f));
        }
        check!(rmdir(dir));
    })

    iotest!(fn file_test_walk_dir() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("walk_dir");
        check!(mkdir(dir, io::UserRWX));

        let dir1 = &dir.join("01/02/03");
        check!(mkdir_recursive(dir1, io::UserRWX));
        check!(File::create(&dir1.join("04")));

        let dir2 = &dir.join("11/12/13");
        check!(mkdir_recursive(dir2, io::UserRWX));
        check!(File::create(&dir2.join("14")));

        let mut files = check!(walk_dir(dir));
        let mut cur = [0u8, .. 2];
        for f in files {
            let stem = f.filestem_str().unwrap();
            let root = stem[0] - ('0' as u8);
            let name = stem[1] - ('0' as u8);
            assert!(cur[root as uint] < name);
            cur[root as uint] = name;
        }

        check!(rmdir_recursive(dir));
    })

    iotest!(fn recursive_mkdir() {
        let tmpdir = tmpdir();
        let dir = tmpdir.join("d1/d2");
        check!(mkdir_recursive(&dir, io::UserRWX));
        assert!(dir.is_dir())
    })

    iotest!(fn recursive_mkdir_slash() {
        check!(mkdir_recursive(&Path::new("/"), io::UserRWX));
    })

    // FIXME(#12795) depends on lstat to work on windows
    #[cfg(not(windows))]
    iotest!(fn recursive_rmdir() {
        let tmpdir = tmpdir();
        let d1 = tmpdir.join("d1");
        let dt = d1.join("t");
        let dtt = dt.join("t");
        let d2 = tmpdir.join("d2");
        let canary = d2.join("do_not_delete");
        check!(mkdir_recursive(&dtt, io::UserRWX));
        check!(mkdir_recursive(&d2, io::UserRWX));
        check!(File::create(&canary).write(bytes!("foo")));
        check!(symlink(&d2, &dt.join("d2")));
        check!(rmdir_recursive(&d1));

        assert!(!d1.is_dir());
        assert!(canary.exists());
    })

    iotest!(fn unicode_path_is_dir() {
        assert!(Path::new(".").is_dir());
        assert!(!Path::new("test/stdtest/fs.rs").is_dir());

        let tmpdir = tmpdir();

        let mut dirpath = tmpdir.path().clone();
        dirpath.push(format!("test-가一ー你好"));
        check!(mkdir(&dirpath, io::UserRWX));
        assert!(dirpath.is_dir());

        let mut filepath = dirpath;
        filepath.push("unicode-file-\uac00\u4e00\u30fc\u4f60\u597d.rs");
        check!(File::create(&filepath)); // ignore return; touch only
        assert!(!filepath.is_dir());
        assert!(filepath.exists());
    })

    iotest!(fn unicode_path_exists() {
        assert!(Path::new(".").exists());
        assert!(!Path::new("test/nonexistent-bogus-path").exists());

        let tmpdir = tmpdir();
        let unicode = tmpdir.path();
        let unicode = unicode.join(format!("test-각丁ー再见"));
        check!(mkdir(&unicode, io::UserRWX));
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

        check!(File::create(&input).write(bytes!("hello")));
        check!(copy(&input, &out));
        let contents = check!(File::open(&out).read_to_end());
        assert_eq!(contents.as_slice(), bytes!("hello"));

        assert_eq!(check!(input.stat()).perm, check!(out.stat()).perm);
    })

    iotest!(fn copy_file_dst_dir() {
        let tmpdir = tmpdir();
        let out = tmpdir.join("out");

        check!(File::create(&out));
        match copy(&out, tmpdir.path()) {
            Ok(..) => fail!(), Err(..) => {}
        }
    })

    iotest!(fn copy_file_dst_exists() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in");
        let output = tmpdir.join("out");

        check!(File::create(&input).write("foo".as_bytes()));
        check!(File::create(&output).write("bar".as_bytes()));
        check!(copy(&input, &output));

        assert_eq!(check!(File::open(&output).read_to_end()),
                   (Vec::from_slice(bytes!("foo"))));
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

        check!(File::create(&input));
        check!(chmod(&input, io::UserRead));
        check!(copy(&input, &out));
        assert!(!check!(out.stat()).perm.intersects(io::UserWrite));

        check!(chmod(&input, io::UserFile));
        check!(chmod(&out, io::UserFile));
    })

    #[cfg(not(windows))] // FIXME(#10264) operation not permitted?
    iotest!(fn symlinks_work() {
        let tmpdir = tmpdir();
        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        check!(File::create(&input).write("foobar".as_bytes()));
        check!(symlink(&input, &out));
        if cfg!(not(windows)) {
            assert_eq!(check!(lstat(&out)).kind, io::TypeSymlink);
            assert_eq!(check!(out.lstat()).kind, io::TypeSymlink);
        }
        assert_eq!(check!(stat(&out)).size, check!(stat(&input)).size);
        assert_eq!(check!(File::open(&out).read_to_end()),
                   (Vec::from_slice(bytes!("foobar"))));
    })

    #[cfg(not(windows))] // apparently windows doesn't like symlinks
    iotest!(fn symlink_noexist() {
        let tmpdir = tmpdir();
        // symlinks can point to things that don't exist
        check!(symlink(&tmpdir.join("foo"), &tmpdir.join("bar")));
        assert!(check!(readlink(&tmpdir.join("bar"))) == tmpdir.join("foo"));
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

        check!(File::create(&input).write("foobar".as_bytes()));
        check!(link(&input, &out));
        if cfg!(not(windows)) {
            assert_eq!(check!(lstat(&out)).kind, io::TypeFile);
            assert_eq!(check!(out.lstat()).kind, io::TypeFile);
            assert_eq!(check!(stat(&out)).unstable.nlink, 2);
            assert_eq!(check!(out.stat()).unstable.nlink, 2);
        }
        assert_eq!(check!(stat(&out)).size, check!(stat(&input)).size);
        assert_eq!(check!(stat(&out)).size, check!(input.stat()).size);
        assert_eq!(check!(File::open(&out).read_to_end()),
                   (Vec::from_slice(bytes!("foobar"))));

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

        check!(File::create(&file));
        assert!(check!(stat(&file)).perm.contains(io::UserWrite));
        check!(chmod(&file, io::UserRead));
        assert!(!check!(stat(&file)).perm.contains(io::UserWrite));

        match chmod(&tmpdir.join("foo"), io::UserRWX) {
            Ok(..) => fail!("wanted a failure"),
            Err(..) => {}
        }

        check!(chmod(&file, io::UserFile));
    })

    iotest!(fn sync_doesnt_kill_anything() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let mut file = check!(File::open_mode(&path, io::Open, io::ReadWrite));
        check!(file.fsync());
        check!(file.datasync());
        check!(file.write(bytes!("foo")));
        check!(file.fsync());
        check!(file.datasync());
        drop(file);
    })

    iotest!(fn truncate_works() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let mut file = check!(File::open_mode(&path, io::Open, io::ReadWrite));
        check!(file.write(bytes!("foo")));
        check!(file.fsync());

        // Do some simple things with truncation
        assert_eq!(check!(file.stat()).size, 3);
        check!(file.truncate(10));
        assert_eq!(check!(file.stat()).size, 10);
        check!(file.write(bytes!("bar")));
        check!(file.fsync());
        assert_eq!(check!(file.stat()).size, 10);
        assert_eq!(check!(File::open(&path).read_to_end()),
                   (Vec::from_slice(bytes!("foobar", 0, 0, 0, 0))));

        // Truncate to a smaller length, don't seek, and then write something.
        // Ensure that the intermediate zeroes are all filled in (we're seeked
        // past the end of the file).
        check!(file.truncate(2));
        assert_eq!(check!(file.stat()).size, 2);
        check!(file.write(bytes!("wut")));
        check!(file.fsync());
        assert_eq!(check!(file.stat()).size, 9);
        assert_eq!(check!(File::open(&path).read_to_end()),
                   (Vec::from_slice(bytes!("fo", 0, 0, 0, 0, "wut"))));
        drop(file);
    })

    iotest!(fn open_flavors() {
        let tmpdir = tmpdir();

        match File::open_mode(&tmpdir.join("a"), io::Open, io::Read) {
            Ok(..) => fail!(), Err(..) => {}
        }

        // Perform each one twice to make sure that it succeeds the second time
        // (where the file exists)
        check!(File::open_mode(&tmpdir.join("b"), io::Open, io::Write));
        assert!(tmpdir.join("b").exists());
        check!(File::open_mode(&tmpdir.join("b"), io::Open, io::Write));

        check!(File::open_mode(&tmpdir.join("c"), io::Open, io::ReadWrite));
        assert!(tmpdir.join("c").exists());
        check!(File::open_mode(&tmpdir.join("c"), io::Open, io::ReadWrite));

        check!(File::open_mode(&tmpdir.join("d"), io::Append, io::Write));
        assert!(tmpdir.join("d").exists());
        check!(File::open_mode(&tmpdir.join("d"), io::Append, io::Write));

        check!(File::open_mode(&tmpdir.join("e"), io::Append, io::ReadWrite));
        assert!(tmpdir.join("e").exists());
        check!(File::open_mode(&tmpdir.join("e"), io::Append, io::ReadWrite));

        check!(File::open_mode(&tmpdir.join("f"), io::Truncate, io::Write));
        assert!(tmpdir.join("f").exists());
        check!(File::open_mode(&tmpdir.join("f"), io::Truncate, io::Write));

        check!(File::open_mode(&tmpdir.join("g"), io::Truncate, io::ReadWrite));
        assert!(tmpdir.join("g").exists());
        check!(File::open_mode(&tmpdir.join("g"), io::Truncate, io::ReadWrite));

        check!(File::create(&tmpdir.join("h")).write("foo".as_bytes()));
        check!(File::open_mode(&tmpdir.join("h"), io::Open, io::Read));
        {
            let mut f = check!(File::open_mode(&tmpdir.join("h"), io::Open,
                                               io::Read));
            match f.write("wut".as_bytes()) {
                Ok(..) => fail!(), Err(..) => {}
            }
        }
        assert!(check!(stat(&tmpdir.join("h"))).size == 3,
                "write/stat failed");
        {
            let mut f = check!(File::open_mode(&tmpdir.join("h"), io::Append,
                                               io::Write));
            check!(f.write("bar".as_bytes()));
        }
        assert!(check!(stat(&tmpdir.join("h"))).size == 6,
                "append didn't append");
        {
            let mut f = check!(File::open_mode(&tmpdir.join("h"), io::Truncate,
                                               io::Write));
            check!(f.write("bar".as_bytes()));
        }
        assert!(check!(stat(&tmpdir.join("h"))).size == 3,
                "truncate didn't truncate");
    })

    #[test]
    fn utime() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("a");
        check!(File::create(&path));

        check!(change_file_times(&path, 1000, 2000));
        assert_eq!(check!(path.stat()).accessed, 1000);
        assert_eq!(check!(path.stat()).modified, 2000);
    }

    #[test]
    fn utime_noexist() {
        let tmpdir = tmpdir();

        match change_file_times(&tmpdir.join("a"), 100, 200) {
            Ok(..) => fail!(),
            Err(..) => {}
        }
    }

    iotest!(fn binary_file() {
        use rand::{StdRng, Rng};

        let mut bytes = [0, ..1024];
        StdRng::new().ok().unwrap().fill_bytes(bytes);

        let tmpdir = tmpdir();

        check!(File::create(&tmpdir.join("test")).write(bytes));
        let actual = check!(File::open(&tmpdir.join("test")).read_to_end());
        assert!(actual.as_slice() == bytes);
    })
}
