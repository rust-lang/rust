//! Filesystem manipulation operations.
//!
//! This module contains basic methods to manipulate the contents of the local
//! filesystem. All methods in this module represent cross-platform filesystem
//! operations. Extra platform-specific functionality can be found in the
//! extension traits of `std::os::$platform`.

#![stable(feature = "rust1", since = "1.0.0")]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(all(
    test,
    not(any(
        target_os = "emscripten",
        target_os = "wasi",
        target_env = "sgx",
        target_os = "xous"
    ))
))]
mod tests;

use crate::ffi::OsString;
use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, Read, Seek, SeekFrom, Write};
use crate::path::{Path, PathBuf};
use crate::sealed::Sealed;
use crate::sync::Arc;
use crate::sys::fs as fs_imp;
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};
use crate::time::SystemTime;

/// An object providing access to an open file on the filesystem.
///
/// An instance of a `File` can be read and/or written depending on what options
/// it was opened with. Files also implement [`Seek`] to alter the logical cursor
/// that the file contains internally.
///
/// Files are automatically closed when they go out of scope.  Errors detected
/// on closing are ignored by the implementation of `Drop`.  Use the method
/// [`sync_all`] if these errors must be manually handled.
///
/// `File` does not buffer reads and writes. For efficiency, consider wrapping the
/// file in a [`BufReader`] or [`BufWriter`] when performing many small [`read`]
/// or [`write`] calls, unless unbuffered reads and writes are required.
///
/// # Examples
///
/// Creates a new file and write bytes to it (you can also use [`write`]):
///
/// ```no_run
/// use std::fs::File;
/// use std::io::prelude::*;
///
/// fn main() -> std::io::Result<()> {
///     let mut file = File::create("foo.txt")?;
///     file.write_all(b"Hello, world!")?;
///     Ok(())
/// }
/// ```
///
/// Reads the contents of a file into a [`String`] (you can also use [`read`]):
///
/// ```no_run
/// use std::fs::File;
/// use std::io::prelude::*;
///
/// fn main() -> std::io::Result<()> {
///     let mut file = File::open("foo.txt")?;
///     let mut contents = String::new();
///     file.read_to_string(&mut contents)?;
///     assert_eq!(contents, "Hello, world!");
///     Ok(())
/// }
/// ```
///
/// Using a buffered [`Read`]er:
///
/// ```no_run
/// use std::fs::File;
/// use std::io::BufReader;
/// use std::io::prelude::*;
///
/// fn main() -> std::io::Result<()> {
///     let file = File::open("foo.txt")?;
///     let mut buf_reader = BufReader::new(file);
///     let mut contents = String::new();
///     buf_reader.read_to_string(&mut contents)?;
///     assert_eq!(contents, "Hello, world!");
///     Ok(())
/// }
/// ```
///
/// Note that, although read and write methods require a `&mut File`, because
/// of the interfaces for [`Read`] and [`Write`], the holder of a `&File` can
/// still modify the file, either through methods that take `&File` or by
/// retrieving the underlying OS object and modifying the file that way.
/// Additionally, many operating systems allow concurrent modification of files
/// by different processes. Avoid assuming that holding a `&File` means that the
/// file will not change.
///
/// # Platform-specific behavior
///
/// On Windows, the implementation of [`Read`] and [`Write`] traits for `File`
/// perform synchronous I/O operations. Therefore the underlying file must not
/// have been opened for asynchronous I/O (e.g. by using `FILE_FLAG_OVERLAPPED`).
///
/// [`BufReader`]: io::BufReader
/// [`BufWriter`]: io::BufWriter
/// [`sync_all`]: File::sync_all
/// [`write`]: File::write
/// [`read`]: File::read
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "File")]
pub struct File {
    inner: fs_imp::File,
}

/// Metadata information about a file.
///
/// This structure is returned from the [`metadata`] or
/// [`symlink_metadata`] function or method and represents known
/// metadata about a file such as its permissions, size, modification
/// times, etc.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Metadata(fs_imp::FileAttr);

/// Iterator over the entries in a directory.
///
/// This iterator is returned from the [`read_dir`] function of this module and
/// will yield instances of <code>[io::Result]<[DirEntry]></code>. Through a [`DirEntry`]
/// information like the entry's path and possibly other metadata can be
/// learned.
///
/// The order in which this iterator returns entries is platform and filesystem
/// dependent.
///
/// # Errors
///
/// This [`io::Result`] will be an [`Err`] if there's some sort of intermittent
/// IO error during iteration.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct ReadDir(fs_imp::ReadDir);

/// Entries returned by the [`ReadDir`] iterator.
///
/// An instance of `DirEntry` represents an entry inside of a directory on the
/// filesystem. Each entry can be inspected via methods to learn about the full
/// path or possibly other metadata through per-platform extension traits.
///
/// # Platform-specific behavior
///
/// On Unix, the `DirEntry` struct contains an internal reference to the open
/// directory. Holding `DirEntry` objects will consume a file handle even
/// after the `ReadDir` iterator is dropped.
///
/// Note that this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
#[stable(feature = "rust1", since = "1.0.0")]
pub struct DirEntry(fs_imp::DirEntry);

/// Options and flags which can be used to configure how a file is opened.
///
/// This builder exposes the ability to configure how a [`File`] is opened and
/// what operations are permitted on the open file. The [`File::open`] and
/// [`File::create`] methods are aliases for commonly used options using this
/// builder.
///
/// Generally speaking, when using `OpenOptions`, you'll first call
/// [`OpenOptions::new`], then chain calls to methods to set each option, then
/// call [`OpenOptions::open`], passing the path of the file you're trying to
/// open. This will give you a [`io::Result`] with a [`File`] inside that you
/// can further operate on.
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
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "FsOpenOptions")]
pub struct OpenOptions(fs_imp::OpenOptions);

/// Representation of the various timestamps on a file.
#[derive(Copy, Clone, Debug, Default)]
#[stable(feature = "file_set_times", since = "1.75.0")]
pub struct FileTimes(fs_imp::FileTimes);

/// Representation of the various permissions on a file.
///
/// This module only currently provides one bit of information,
/// [`Permissions::readonly`], which is exposed on all currently supported
/// platforms. Unix-specific functionality, such as mode bits, is available
/// through the [`PermissionsExt`] trait.
///
/// [`PermissionsExt`]: crate::os::unix::fs::PermissionsExt
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "FsPermissions")]
pub struct Permissions(fs_imp::FilePermissions);

/// A structure representing a type of file with accessors for each file type.
/// It is returned by [`Metadata::file_type`] method.
#[stable(feature = "file_type", since = "1.1.0")]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(not(test), rustc_diagnostic_item = "FileType")]
pub struct FileType(fs_imp::FileType);

/// A builder used to create directories in various manners.
///
/// This builder also supports platform-specific options.
#[stable(feature = "dir_builder", since = "1.6.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "DirBuilder")]
#[derive(Debug)]
pub struct DirBuilder {
    inner: fs_imp::DirBuilder,
    recursive: bool,
}

/// Reads the entire contents of a file into a bytes vector.
///
/// This is a convenience function for using [`File::open`] and [`read_to_end`]
/// with fewer imports and without an intermediate variable.
///
/// [`read_to_end`]: Read::read_to_end
///
/// # Errors
///
/// This function will return an error if `path` does not already exist.
/// Other errors may also be returned according to [`OpenOptions::open`].
///
/// While reading from the file, this function handles [`io::ErrorKind::Interrupted`]
/// with automatic retries. See [io::Read] documentation for details.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
///     let data: Vec<u8> = fs::read("image.jpg")?;
///     assert_eq!(data[0..3], [0xFF, 0xD8, 0xFF]);
///     Ok(())
/// }
/// ```
#[stable(feature = "fs_read_write_bytes", since = "1.26.0")]
pub fn read<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    fn inner(path: &Path) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let size = file.metadata().map(|m| m.len() as usize).ok();
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size.unwrap_or(0))?;
        io::default_read_to_end(&mut file, &mut bytes, size)?;
        Ok(bytes)
    }
    inner(path.as_ref())
}

/// Reads the entire contents of a file into a string.
///
/// This is a convenience function for using [`File::open`] and [`read_to_string`]
/// with fewer imports and without an intermediate variable.
///
/// [`read_to_string`]: Read::read_to_string
///
/// # Errors
///
/// This function will return an error if `path` does not already exist.
/// Other errors may also be returned according to [`OpenOptions::open`].
///
/// If the contents of the file are not valid UTF-8, then an error will also be
/// returned.
///
/// While reading from the file, this function handles [`io::ErrorKind::Interrupted`]
/// with automatic retries. See [io::Read] documentation for details.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
/// use std::error::Error;
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let message: String = fs::read_to_string("message.txt")?;
///     println!("{}", message);
///     Ok(())
/// }
/// ```
#[stable(feature = "fs_read_write", since = "1.26.0")]
pub fn read_to_string<P: AsRef<Path>>(path: P) -> io::Result<String> {
    fn inner(path: &Path) -> io::Result<String> {
        let mut file = File::open(path)?;
        let size = file.metadata().map(|m| m.len() as usize).ok();
        let mut string = String::new();
        string.try_reserve_exact(size.unwrap_or(0))?;
        io::default_read_to_string(&mut file, &mut string, size)?;
        Ok(string)
    }
    inner(path.as_ref())
}

/// Writes a slice as the entire contents of a file.
///
/// This function will create a file if it does not exist,
/// and will entirely replace its contents if it does.
///
/// Depending on the platform, this function may fail if the
/// full directory path does not exist.
///
/// This is a convenience function for using [`File::create`] and [`write_all`]
/// with fewer imports.
///
/// [`write_all`]: Write::write_all
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::write("foo.txt", b"Lorem ipsum")?;
///     fs::write("bar.txt", "dolor sit")?;
///     Ok(())
/// }
/// ```
#[stable(feature = "fs_read_write_bytes", since = "1.26.0")]
pub fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    fn inner(path: &Path, contents: &[u8]) -> io::Result<()> {
        File::create(path)?.write_all(contents)
    }
    inner(path.as_ref(), contents.as_ref())
}

impl File {
    /// Attempts to open a file in read-only mode.
    ///
    /// See the [`OpenOptions::open`] method for more details.
    ///
    /// If you only need to read the entire file contents,
    /// consider [`std::fs::read()`][self::read] or
    /// [`std::fs::read_to_string()`][self::read_to_string] instead.
    ///
    /// # Errors
    ///
    /// This function will return an error if `path` does not already exist.
    /// Other errors may also be returned according to [`OpenOptions::open`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::Read;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::open("foo.txt")?;
    ///     let mut data = vec![];
    ///     f.read_to_end(&mut data)?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<File> {
        OpenOptions::new().read(true).open(path.as_ref())
    }

    /// Attempts to open a file in read-only mode with buffering.
    ///
    /// See the [`OpenOptions::open`] method, the [`BufReader`][io::BufReader] type,
    /// and the [`BufRead`][io::BufRead] trait for more details.
    ///
    /// If you only need to read the entire file contents,
    /// consider [`std::fs::read()`][self::read] or
    /// [`std::fs::read_to_string()`][self::read_to_string] instead.
    ///
    /// # Errors
    ///
    /// This function will return an error if `path` does not already exist,
    /// or if memory allocation fails for the new buffer.
    /// Other errors may also be returned according to [`OpenOptions::open`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(file_buffered)]
    /// use std::fs::File;
    /// use std::io::BufRead;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::open_buffered("foo.txt")?;
    ///     assert!(f.capacity() > 0);
    ///     for (line, i) in f.lines().zip(1..) {
    ///         println!("{i:6}: {}", line?);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "file_buffered", issue = "130804")]
    pub fn open_buffered<P: AsRef<Path>>(path: P) -> io::Result<io::BufReader<File>> {
        // Allocate the buffer *first* so we don't affect the filesystem otherwise.
        let buffer = io::BufReader::<Self>::try_new_buffer()?;
        let file = File::open(path)?;
        Ok(io::BufReader::with_buffer(file, buffer))
    }

    /// Opens a file in write-only mode.
    ///
    /// This function will create a file if it does not exist,
    /// and will truncate it if it does.
    ///
    /// Depending on the platform, this function may fail if the
    /// full directory path does not exist.
    /// See the [`OpenOptions::open`] function for more details.
    ///
    /// See also [`std::fs::write()`][self::write] for a simple function to
    /// create a file with some given data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::Write;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::create("foo.txt")?;
    ///     f.write_all(&1234_u32.to_be_bytes())?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<File> {
        OpenOptions::new().write(true).create(true).truncate(true).open(path.as_ref())
    }

    /// Opens a file in write-only mode with buffering.
    ///
    /// This function will create a file if it does not exist,
    /// and will truncate it if it does.
    ///
    /// Depending on the platform, this function may fail if the
    /// full directory path does not exist.
    ///
    /// See the [`OpenOptions::open`] method and the
    /// [`BufWriter`][io::BufWriter] type for more details.
    ///
    /// See also [`std::fs::write()`][self::write] for a simple function to
    /// create a file with some given data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(file_buffered)]
    /// use std::fs::File;
    /// use std::io::Write;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::create_buffered("foo.txt")?;
    ///     assert!(f.capacity() > 0);
    ///     for i in 0..100 {
    ///         writeln!(&mut f, "{i}")?;
    ///     }
    ///     f.flush()?;
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "file_buffered", issue = "130804")]
    pub fn create_buffered<P: AsRef<Path>>(path: P) -> io::Result<io::BufWriter<File>> {
        // Allocate the buffer *first* so we don't affect the filesystem otherwise.
        let buffer = io::BufWriter::<Self>::try_new_buffer()?;
        let file = File::create(path)?;
        Ok(io::BufWriter::with_buffer(file, buffer))
    }

    /// Creates a new file in read-write mode; error if the file exists.
    ///
    /// This function will create a file if it does not exist, or return an error if it does. This
    /// way, if the call succeeds, the file returned is guaranteed to be new.
    /// If a file exists at the target location, creating a new file will fail with [`AlreadyExists`]
    /// or another error based on the situation. See [`OpenOptions::open`] for a
    /// non-exhaustive list of likely errors.
    ///
    /// This option is useful because it is atomic. Otherwise between checking whether a file
    /// exists and creating a new one, the file may have been created by another process (a TOCTOU
    /// race condition / attack).
    ///
    /// This can also be written using
    /// `File::options().read(true).write(true).create_new(true).open(...)`.
    ///
    /// [`AlreadyExists`]: crate::io::ErrorKind::AlreadyExists
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::Write;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::create_new("foo.txt")?;
    ///     f.write_all("Hello, world!".as_bytes())?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "file_create_new", since = "1.77.0")]
    pub fn create_new<P: AsRef<Path>>(path: P) -> io::Result<File> {
        OpenOptions::new().read(true).write(true).create_new(true).open(path.as_ref())
    }

    /// Returns a new OpenOptions object.
    ///
    /// This function returns a new OpenOptions object that you can use to
    /// open or create a file with specific options if `open()` or `create()`
    /// are not appropriate.
    ///
    /// It is equivalent to `OpenOptions::new()`, but allows you to write more
    /// readable code. Instead of
    /// `OpenOptions::new().append(true).open("example.log")`,
    /// you can write `File::options().append(true).open("example.log")`. This
    /// also avoids the need to import `OpenOptions`.
    ///
    /// See the [`OpenOptions::new`] function for more details.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::Write;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::options().append(true).open("example.log")?;
    ///     writeln!(&mut f, "new line")?;
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "with_options", since = "1.58.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "file_options")]
    pub fn options() -> OpenOptions {
        OpenOptions::new()
    }

    /// Attempts to sync all OS-internal file content and metadata to disk.
    ///
    /// This function will attempt to ensure that all in-memory data reaches the
    /// filesystem before returning.
    ///
    /// This can be used to handle errors that would otherwise only be caught
    /// when the `File` is closed, as dropping a `File` will ignore all errors.
    /// Note, however, that `sync_all` is generally more expensive than closing
    /// a file by dropping it, because the latter is not required to block until
    /// the data has been written to the filesystem.
    ///
    /// If synchronizing the metadata is not required, use [`sync_data`] instead.
    ///
    /// [`sync_data`]: File::sync_data
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::prelude::*;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::create("foo.txt")?;
    ///     f.write_all(b"Hello, world!")?;
    ///
    ///     f.sync_all()?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[doc(alias = "fsync")]
    pub fn sync_all(&self) -> io::Result<()> {
        self.inner.fsync()
    }

    /// This function is similar to [`sync_all`], except that it might not
    /// synchronize file metadata to the filesystem.
    ///
    /// This is intended for use cases that must synchronize content, but don't
    /// need the metadata on disk. The goal of this method is to reduce disk
    /// operations.
    ///
    /// Note that some platforms may simply implement this in terms of
    /// [`sync_all`].
    ///
    /// [`sync_all`]: File::sync_all
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::prelude::*;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::create("foo.txt")?;
    ///     f.write_all(b"Hello, world!")?;
    ///
    ///     f.sync_data()?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[doc(alias = "fdatasync")]
    pub fn sync_data(&self) -> io::Result<()> {
        self.inner.datasync()
    }

    /// Acquire an exclusive lock on the file. Blocks until the lock can be acquired.
    ///
    /// This acquires an exclusive lock; no other file handle to this file may acquire another lock.
    ///
    /// If this file handle/descriptor, or a clone of it, already holds an lock the exact behavior
    /// is unspecified and platform dependent, including the possibility that it will deadlock.
    /// However, if this method returns, then an exclusive lock is held.
    ///
    /// If the file not open for writing, it is unspecified whether this function returns an error.
    ///
    /// This lock may be advisory or mandatory. This lock is meant to interact with [`lock`],
    /// [`try_lock`], [`lock_shared`], [`try_lock_shared`], and [`unlock`]. Its interactions with
    /// other methods, such as [`read`] and [`write`] are platform specific, and it may or may not
    /// cause non-lockholders to block.
    ///
    /// The lock will be released when this file (along with any other file descriptors/handles
    /// duplicated or inherited from it) is closed, or if the [`unlock`] method is called.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `flock` function on Unix with the `LOCK_EX` flag,
    /// and the `LockFileEx` function on Windows with the `LOCKFILE_EXCLUSIVE_LOCK` flag. Note that,
    /// this [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    ///
    /// [`lock`]: File::lock
    /// [`lock_shared`]: File::lock_shared
    /// [`try_lock`]: File::try_lock
    /// [`try_lock_shared`]: File::try_lock_shared
    /// [`unlock`]: File::unlock
    /// [`read`]: Read::read
    /// [`write`]: Write::write
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(file_lock)]
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::create("foo.txt")?;
    ///     f.lock()?;
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "file_lock", issue = "130994")]
    pub fn lock(&self) -> io::Result<()> {
        self.inner.lock()
    }

    /// Acquire a shared (non-exclusive) lock on the file. Blocks until the lock can be acquired.
    ///
    /// This acquires a shared lock; more than one file handle may hold a shared lock, but none may
    /// hold an exclusive lock at the same time.
    ///
    /// If this file handle/descriptor, or a clone of it, already holds an lock, the exact behavior
    /// is unspecified and platform dependent, including the possibility that it will deadlock.
    /// However, if this method returns, then a shared lock is held.
    ///
    /// This lock may be advisory or mandatory. This lock is meant to interact with [`lock`],
    /// [`try_lock`], [`lock_shared`], [`try_lock_shared`], and [`unlock`]. Its interactions with
    /// other methods, such as [`read`] and [`write`] are platform specific, and it may or may not
    /// cause non-lockholders to block.
    ///
    /// The lock will be released when this file (along with any other file descriptors/handles
    /// duplicated or inherited from it) is closed, or if the [`unlock`] method is called.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `flock` function on Unix with the `LOCK_SH` flag,
    /// and the `LockFileEx` function on Windows. Note that, this
    /// [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    ///
    /// [`lock`]: File::lock
    /// [`lock_shared`]: File::lock_shared
    /// [`try_lock`]: File::try_lock
    /// [`try_lock_shared`]: File::try_lock_shared
    /// [`unlock`]: File::unlock
    /// [`read`]: Read::read
    /// [`write`]: Write::write
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(file_lock)]
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("foo.txt")?;
    ///     f.lock_shared()?;
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "file_lock", issue = "130994")]
    pub fn lock_shared(&self) -> io::Result<()> {
        self.inner.lock_shared()
    }

    /// Try to acquire an exclusive lock on the file.
    ///
    /// Returns `Ok(false)` if a different lock is already held on this file (via another
    /// handle/descriptor).
    ///
    /// This acquires an exclusive lock; no other file handle to this file may acquire another lock.
    ///
    /// If this file handle/descriptor, or a clone of it, already holds an lock, the exact behavior
    /// is unspecified and platform dependent, including the possibility that it will deadlock.
    /// However, if this method returns `Ok(true)`, then it has acquired an exclusive lock.
    ///
    /// If the file not open for writing, it is unspecified whether this function returns an error.
    ///
    /// This lock may be advisory or mandatory. This lock is meant to interact with [`lock`],
    /// [`try_lock`], [`lock_shared`], [`try_lock_shared`], and [`unlock`]. Its interactions with
    /// other methods, such as [`read`] and [`write`] are platform specific, and it may or may not
    /// cause non-lockholders to block.
    ///
    /// The lock will be released when this file (along with any other file descriptors/handles
    /// duplicated or inherited from it) is closed, or if the [`unlock`] method is called.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `flock` function on Unix with the `LOCK_EX` and
    /// `LOCK_NB` flags, and the `LockFileEx` function on Windows with the `LOCKFILE_EXCLUSIVE_LOCK`
    /// and `LOCKFILE_FAIL_IMMEDIATELY` flags. Note that, this
    /// [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    ///
    /// [`lock`]: File::lock
    /// [`lock_shared`]: File::lock_shared
    /// [`try_lock`]: File::try_lock
    /// [`try_lock_shared`]: File::try_lock_shared
    /// [`unlock`]: File::unlock
    /// [`read`]: Read::read
    /// [`write`]: Write::write
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(file_lock)]
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::create("foo.txt")?;
    ///     f.try_lock()?;
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "file_lock", issue = "130994")]
    pub fn try_lock(&self) -> io::Result<bool> {
        self.inner.try_lock()
    }

    /// Try to acquire a shared (non-exclusive) lock on the file.
    ///
    /// Returns `Ok(false)` if an exclusive lock is already held on this file (via another
    /// handle/descriptor).
    ///
    /// This acquires a shared lock; more than one file handle may hold a shared lock, but none may
    /// hold an exclusive lock at the same time.
    ///
    /// If this file handle, or a clone of it, already holds an lock, the exact behavior is
    /// unspecified and platform dependent, including the possibility that it will deadlock.
    /// However, if this method returns `Ok(true)`, then it has acquired a shared lock.
    ///
    /// This lock may be advisory or mandatory. This lock is meant to interact with [`lock`],
    /// [`try_lock`], [`lock_shared`], [`try_lock_shared`], and [`unlock`]. Its interactions with
    /// other methods, such as [`read`] and [`write`] are platform specific, and it may or may not
    /// cause non-lockholders to block.
    ///
    /// The lock will be released when this file (along with any other file descriptors/handles
    /// duplicated or inherited from it) is closed, or if the [`unlock`] method is called.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `flock` function on Unix with the `LOCK_SH` and
    /// `LOCK_NB` flags, and the `LockFileEx` function on Windows with the
    /// `LOCKFILE_FAIL_IMMEDIATELY` flag. Note that, this
    /// [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    ///
    /// [`lock`]: File::lock
    /// [`lock_shared`]: File::lock_shared
    /// [`try_lock`]: File::try_lock
    /// [`try_lock_shared`]: File::try_lock_shared
    /// [`unlock`]: File::unlock
    /// [`read`]: Read::read
    /// [`write`]: Write::write
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(file_lock)]
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("foo.txt")?;
    ///     f.try_lock_shared()?;
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "file_lock", issue = "130994")]
    pub fn try_lock_shared(&self) -> io::Result<bool> {
        self.inner.try_lock_shared()
    }

    /// Release all locks on the file.
    ///
    /// All locks are released when the file (along with any other file descriptors/handles
    /// duplicated or inherited from it) is closed. This method allows releasing locks without
    /// closing the file.
    ///
    /// If no lock is currently held via this file descriptor/handle, this method may return an
    /// error, or may return successfully without taking any action.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `flock` function on Unix with the `LOCK_UN` flag,
    /// and the `UnlockFile` function on Windows. Note that, this
    /// [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(file_lock)]
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("foo.txt")?;
    ///     f.lock()?;
    ///     f.unlock()?;
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "file_lock", issue = "130994")]
    pub fn unlock(&self) -> io::Result<()> {
        self.inner.unlock()
    }

    /// Truncates or extends the underlying file, updating the size of
    /// this file to become `size`.
    ///
    /// If the `size` is less than the current file's size, then the file will
    /// be shrunk. If it is greater than the current file's size, then the file
    /// will be extended to `size` and have all of the intermediate data filled
    /// in with 0s.
    ///
    /// The file's cursor isn't changed. In particular, if the cursor was at the
    /// end and the file is shrunk using this operation, the cursor will now be
    /// past the end.
    ///
    /// # Errors
    ///
    /// This function will return an error if the file is not opened for writing.
    /// Also, [`std::io::ErrorKind::InvalidInput`](crate::io::ErrorKind::InvalidInput)
    /// will be returned if the desired length would cause an overflow due to
    /// the implementation specifics.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::create("foo.txt")?;
    ///     f.set_len(10)?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// Note that this method alters the content of the underlying file, even
    /// though it takes `&self` rather than `&mut self`.
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
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::open("foo.txt")?;
    ///     let metadata = f.metadata()?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn metadata(&self) -> io::Result<Metadata> {
        self.inner.file_attr().map(Metadata)
    }

    /// Creates a new `File` instance that shares the same underlying file handle
    /// as the existing `File` instance. Reads, writes, and seeks will affect
    /// both `File` instances simultaneously.
    ///
    /// # Examples
    ///
    /// Creates two handles for a file named `foo.txt`:
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut file = File::open("foo.txt")?;
    ///     let file_copy = file.try_clone()?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// Assuming there’s a file named `foo.txt` with contents `abcdef\n`, create
    /// two handles, seek one of them, and read the remaining bytes from the
    /// other handle:
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::SeekFrom;
    /// use std::io::prelude::*;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut file = File::open("foo.txt")?;
    ///     let mut file_copy = file.try_clone()?;
    ///
    ///     file.seek(SeekFrom::Start(3))?;
    ///
    ///     let mut contents = vec![];
    ///     file_copy.read_to_end(&mut contents)?;
    ///     assert_eq!(contents, b"def\n");
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "file_try_clone", since = "1.9.0")]
    pub fn try_clone(&self) -> io::Result<File> {
        Ok(File { inner: self.inner.duplicate()? })
    }

    /// Changes the permissions on the underlying file.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `fchmod` function on Unix and
    /// the `SetFileInformationByHandle` function on Windows. Note that, this
    /// [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    ///
    /// # Errors
    ///
    /// This function will return an error if the user lacks permission change
    /// attributes on the underlying file. It may also return an error in other
    /// os-specific unspecified cases.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// fn main() -> std::io::Result<()> {
    ///     use std::fs::File;
    ///
    ///     let file = File::open("foo.txt")?;
    ///     let mut perms = file.metadata()?.permissions();
    ///     perms.set_readonly(true);
    ///     file.set_permissions(perms)?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// Note that this method alters the permissions of the underlying file,
    /// even though it takes `&self` rather than `&mut self`.
    #[doc(alias = "fchmod", alias = "SetFileInformationByHandle")]
    #[stable(feature = "set_permissions_atomic", since = "1.16.0")]
    pub fn set_permissions(&self, perm: Permissions) -> io::Result<()> {
        self.inner.set_permissions(perm.0)
    }

    /// Changes the timestamps of the underlying file.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `futimens` function on Unix (falling back to
    /// `futimes` on macOS before 10.13) and the `SetFileTime` function on Windows. Note that this
    /// [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    ///
    /// # Errors
    ///
    /// This function will return an error if the user lacks permission to change timestamps on the
    /// underlying file. It may also return an error in other os-specific unspecified cases.
    ///
    /// This function may return an error if the operating system lacks support to change one or
    /// more of the timestamps set in the `FileTimes` structure.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// fn main() -> std::io::Result<()> {
    ///     use std::fs::{self, File, FileTimes};
    ///
    ///     let src = fs::metadata("src")?;
    ///     let dest = File::options().write(true).open("dest")?;
    ///     let times = FileTimes::new()
    ///         .set_accessed(src.accessed()?)
    ///         .set_modified(src.modified()?);
    ///     dest.set_times(times)?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "file_set_times", since = "1.75.0")]
    #[doc(alias = "futimens")]
    #[doc(alias = "futimes")]
    #[doc(alias = "SetFileTime")]
    pub fn set_times(&self, times: FileTimes) -> io::Result<()> {
        self.inner.set_times(times.0)
    }

    /// Changes the modification time of the underlying file.
    ///
    /// This is an alias for `set_times(FileTimes::new().set_modified(time))`.
    #[stable(feature = "file_set_times", since = "1.75.0")]
    #[inline]
    pub fn set_modified(&self, time: SystemTime) -> io::Result<()> {
        self.set_times(FileTimes::new().set_modified(time))
    }
}

// In addition to the `impl`s here, `File` also has `impl`s for
// `AsFd`/`From<OwnedFd>`/`Into<OwnedFd>` and
// `AsRawFd`/`IntoRawFd`/`FromRawFd`, on Unix and WASI, and
// `AsHandle`/`From<OwnedHandle>`/`Into<OwnedHandle>` and
// `AsRawHandle`/`IntoRawHandle`/`FromRawHandle` on Windows.

impl AsInner<fs_imp::File> for File {
    #[inline]
    fn as_inner(&self) -> &fs_imp::File {
        &self.inner
    }
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

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

/// Indicates how much extra capacity is needed to read the rest of the file.
fn buffer_capacity_required(mut file: &File) -> Option<usize> {
    let size = file.metadata().map(|m| m.len()).ok()?;
    let pos = file.stream_position().ok()?;
    // Don't worry about `usize` overflow because reading will fail regardless
    // in that case.
    Some(size.saturating_sub(pos) as usize)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for &File {
    /// Reads some bytes from the file.
    ///
    /// See [`Read::read`] docs for more info.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `read` function on Unix and
    /// the `NtReadFile` function on Windows. Note that this [may change in
    /// the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    /// Like `read`, except that it reads into a slice of buffers.
    ///
    /// See [`Read::read_vectored`] docs for more info.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `readv` function on Unix and
    /// falls back to the `read` implementation on Windows. Note that this
    /// [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    #[inline]
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    fn read_buf(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.inner.read_buf(cursor)
    }

    /// Determines if `File` has an efficient `read_vectored` implementation.
    ///
    /// See [`Read::is_read_vectored`] docs for more info.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently returns `true` on Unix an `false` on Windows.
    /// Note that this [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.inner.is_read_vectored()
    }

    // Reserves space in the buffer based on the file size when available.
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let size = buffer_capacity_required(self);
        buf.try_reserve(size.unwrap_or(0))?;
        io::default_read_to_end(self, buf, size)
    }

    // Reserves space in the buffer based on the file size when available.
    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        let size = buffer_capacity_required(self);
        buf.try_reserve(size.unwrap_or(0))?;
        io::default_read_to_string(self, buf, size)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for &File {
    /// Writes some bytes to the file.
    ///
    /// See [`Write::write`] docs for more info.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `write` function on Unix and
    /// the `NtWriteFile` function on Windows. Note that this [may change in
    /// the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    /// Like `write`, except that it writes into a slice of buffers.
    ///
    /// See [`Write::write_vectored`] docs for more info.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `writev` function on Unix
    /// and falls back to the `write` implementation on Windows. Note that this
    /// [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.write_vectored(bufs)
    }

    /// Determines if `File` has an efficient `write_vectored` implementation.
    ///
    /// See [`Write::is_write_vectored`] docs for more info.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently returns `true` on Unix an `false` on Windows.
    /// Note that this [may change in the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    /// Flushes the file, ensuring that all intermediately buffered contents
    /// reach their destination.
    ///
    /// See [`Write::flush`] docs for more info.
    ///
    /// # Platform-specific behavior
    ///
    /// Since a `File` structure doesn't contain any buffers, this function is
    /// currently a no-op on Unix and Windows. Note that this [may change in
    /// the future][changes].
    ///
    /// [changes]: io#platform-specific-behavior
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Seek for &File {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for File {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (&*self).read(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        (&*self).read_vectored(bufs)
    }
    fn read_buf(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        (&*self).read_buf(cursor)
    }
    #[inline]
    fn is_read_vectored(&self) -> bool {
        (&&*self).is_read_vectored()
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        (&*self).read_to_end(buf)
    }
    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        (&*self).read_to_string(buf)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for File {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&*self).write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        (&*self).write_vectored(bufs)
    }
    #[inline]
    fn is_write_vectored(&self) -> bool {
        (&&*self).is_write_vectored()
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        (&*self).flush()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Seek for File {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        (&*self).seek(pos)
    }
}

#[stable(feature = "io_traits_arc", since = "1.73.0")]
impl Read for Arc<File> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (&**self).read(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        (&**self).read_vectored(bufs)
    }
    fn read_buf(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        (&**self).read_buf(cursor)
    }
    #[inline]
    fn is_read_vectored(&self) -> bool {
        (&**self).is_read_vectored()
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        (&**self).read_to_end(buf)
    }
    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        (&**self).read_to_string(buf)
    }
}
#[stable(feature = "io_traits_arc", since = "1.73.0")]
impl Write for Arc<File> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&**self).write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        (&**self).write_vectored(bufs)
    }
    #[inline]
    fn is_write_vectored(&self) -> bool {
        (&**self).is_write_vectored()
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        (&**self).flush()
    }
}
#[stable(feature = "io_traits_arc", since = "1.73.0")]
impl Seek for Arc<File> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        (&**self).seek(pos)
    }
}

impl OpenOptions {
    /// Creates a blank new set of options ready for configuration.
    ///
    /// All options are initially set to `false`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let mut options = OpenOptions::new();
    /// let file = options.read(true).open("foo.txt");
    /// ```
    #[cfg_attr(not(test), rustc_diagnostic_item = "open_options_new")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    pub fn new() -> Self {
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
    pub fn read(&mut self, read: bool) -> &mut Self {
        self.0.read(read);
        self
    }

    /// Sets the option for write access.
    ///
    /// This option, when true, will indicate that the file should be
    /// `write`-able if opened.
    ///
    /// If the file already exists, any write calls on it will overwrite its
    /// contents, without truncating it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().write(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write(&mut self, write: bool) -> &mut Self {
        self.0.write(write);
        self
    }

    /// Sets the option for the append mode.
    ///
    /// This option, when true, means that writes will append to a file instead
    /// of overwriting previous contents.
    /// Note that setting `.write(true).append(true)` has the same effect as
    /// setting only `.append(true)`.
    ///
    /// Append mode guarantees that writes will be positioned at the current end of file,
    /// even when there are other processes or threads appending to the same file. This is
    /// unlike <code>[seek]\([SeekFrom]::[End]\(0))</code> followed by `write()`, which
    /// has a race between seeking and writing during which another writer can write, with
    /// our `write()` overwriting their data.
    ///
    /// Keep in mind that this does not necessarily guarantee that data appended by
    /// different processes or threads does not interleave. The amount of data accepted a
    /// single `write()` call depends on the operating system and file system. A
    /// successful `write()` is allowed to write only part of the given data, so even if
    /// you're careful to provide the whole message in a single call to `write()`, there
    /// is no guarantee that it will be written out in full. If you rely on the filesystem
    /// accepting the message in a single write, make sure that all data that belongs
    /// together is written in one operation. This can be done by concatenating strings
    /// before passing them to [`write()`].
    ///
    /// If a file is opened with both read and append access, beware that after
    /// opening, and after every write, the position for reading may be set at the
    /// end of the file. So, before writing, save the current position (using
    /// <code>[Seek]::[stream_position]</code>), and restore it before the next read.
    ///
    /// ## Note
    ///
    /// This function doesn't create the file if it doesn't exist. Use the
    /// [`OpenOptions::create`] method to do so.
    ///
    /// [`write()`]: Write::write "io::Write::write"
    /// [`flush()`]: Write::flush "io::Write::flush"
    /// [stream_position]: Seek::stream_position "io::Seek::stream_position"
    /// [seek]: Seek::seek "io::Seek::seek"
    /// [Current]: SeekFrom::Current "io::SeekFrom::Current"
    /// [End]: SeekFrom::End "io::SeekFrom::End"
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().append(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn append(&mut self, append: bool) -> &mut Self {
        self.0.append(append);
        self
    }

    /// Sets the option for truncating a previous file.
    ///
    /// If a file is successfully opened with this option set to true, it will truncate
    /// the file to 0 length if it already exists.
    ///
    /// The file must be opened with write access for truncate to work.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().write(true).truncate(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, truncate: bool) -> &mut Self {
        self.0.truncate(truncate);
        self
    }

    /// Sets the option to create a new file, or open it if it already exists.
    ///
    /// In order for the file to be created, [`OpenOptions::write`] or
    /// [`OpenOptions::append`] access must be used.
    ///
    /// See also [`std::fs::write()`][self::write] for a simple function to
    /// create a file with some given data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().write(true).create(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn create(&mut self, create: bool) -> &mut Self {
        self.0.create(create);
        self
    }

    /// Sets the option to create a new file, failing if it already exists.
    ///
    /// No file is allowed to exist at the target location, also no (dangling) symlink. In this
    /// way, if the call succeeds, the file returned is guaranteed to be new.
    /// If a file exists at the target location, creating a new file will fail with [`AlreadyExists`]
    /// or another error based on the situation. See [`OpenOptions::open`] for a
    /// non-exhaustive list of likely errors.
    ///
    /// This option is useful because it is atomic. Otherwise between checking
    /// whether a file exists and creating a new one, the file may have been
    /// created by another process (a TOCTOU race condition / attack).
    ///
    /// If `.create_new(true)` is set, [`.create()`] and [`.truncate()`] are
    /// ignored.
    ///
    /// The file must be opened with write or append access in order to create
    /// a new file.
    ///
    /// [`.create()`]: OpenOptions::create
    /// [`.truncate()`]: OpenOptions::truncate
    /// [`AlreadyExists`]: io::ErrorKind::AlreadyExists
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().write(true)
    ///                              .create_new(true)
    ///                              .open("foo.txt");
    /// ```
    #[stable(feature = "expand_open_options2", since = "1.9.0")]
    pub fn create_new(&mut self, create_new: bool) -> &mut Self {
        self.0.create_new(create_new);
        self
    }

    /// Opens a file at `path` with the options specified by `self`.
    ///
    /// # Errors
    ///
    /// This function will return an error under a number of different
    /// circumstances. Some of these error conditions are listed here, together
    /// with their [`io::ErrorKind`]. The mapping to [`io::ErrorKind`]s is not
    /// part of the compatibility contract of the function.
    ///
    /// * [`NotFound`]: The specified file does not exist and neither `create`
    ///   or `create_new` is set.
    /// * [`NotFound`]: One of the directory components of the file path does
    ///   not exist.
    /// * [`PermissionDenied`]: The user lacks permission to get the specified
    ///   access rights for the file.
    /// * [`PermissionDenied`]: The user lacks permission to open one of the
    ///   directory components of the specified path.
    /// * [`AlreadyExists`]: `create_new` was specified and the file already
    ///   exists.
    /// * [`InvalidInput`]: Invalid combinations of open options (truncate
    ///   without write access, no access mode set, etc.).
    ///
    /// The following errors don't match any existing [`io::ErrorKind`] at the moment:
    /// * One of the directory components of the specified file path
    ///   was not, in fact, a directory.
    /// * Filesystem-level errors: full disk, write permission
    ///   requested on a read-only file system, exceeded disk quota, too many
    ///   open files, too long filename, too many symbolic links in the
    ///   specified path (Unix-like systems only), etc.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().read(true).open("foo.txt");
    /// ```
    ///
    /// [`AlreadyExists`]: io::ErrorKind::AlreadyExists
    /// [`InvalidInput`]: io::ErrorKind::InvalidInput
    /// [`NotFound`]: io::ErrorKind::NotFound
    /// [`PermissionDenied`]: io::ErrorKind::PermissionDenied
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn open<P: AsRef<Path>>(&self, path: P) -> io::Result<File> {
        self._open(path.as_ref())
    }

    fn _open(&self, path: &Path) -> io::Result<File> {
        fs_imp::File::open(path, &self.0).map(|inner| File { inner })
    }
}

impl AsInner<fs_imp::OpenOptions> for OpenOptions {
    #[inline]
    fn as_inner(&self) -> &fs_imp::OpenOptions {
        &self.0
    }
}

impl AsInnerMut<fs_imp::OpenOptions> for OpenOptions {
    #[inline]
    fn as_inner_mut(&mut self) -> &mut fs_imp::OpenOptions {
        &mut self.0
    }
}

impl Metadata {
    /// Returns the file type for this metadata.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// fn main() -> std::io::Result<()> {
    ///     use std::fs;
    ///
    ///     let metadata = fs::metadata("foo.txt")?;
    ///
    ///     println!("{:?}", metadata.file_type());
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn file_type(&self) -> FileType {
        FileType(self.0.file_type())
    }

    /// Returns `true` if this metadata is for a directory. The
    /// result is mutually exclusive to the result of
    /// [`Metadata::is_file`], and will be false for symlink metadata
    /// obtained from [`symlink_metadata`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// fn main() -> std::io::Result<()> {
    ///     use std::fs;
    ///
    ///     let metadata = fs::metadata("foo.txt")?;
    ///
    ///     assert!(!metadata.is_dir());
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_dir(&self) -> bool {
        self.file_type().is_dir()
    }

    /// Returns `true` if this metadata is for a regular file. The
    /// result is mutually exclusive to the result of
    /// [`Metadata::is_dir`], and will be false for symlink metadata
    /// obtained from [`symlink_metadata`].
    ///
    /// When the goal is simply to read from (or write to) the source, the most
    /// reliable way to test the source can be read (or written to) is to open
    /// it. Only using `is_file` can break workflows like `diff <( prog_a )` on
    /// a Unix-like system for example. See [`File::open`] or
    /// [`OpenOptions::open`] for more information.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let metadata = fs::metadata("foo.txt")?;
    ///
    ///     assert!(metadata.is_file());
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_file(&self) -> bool {
        self.file_type().is_file()
    }

    /// Returns `true` if this metadata is for a symbolic link.
    ///
    /// # Examples
    ///
    #[cfg_attr(unix, doc = "```no_run")]
    #[cfg_attr(not(unix), doc = "```ignore")]
    /// use std::fs;
    /// use std::path::Path;
    /// use std::os::unix::fs::symlink;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let link_path = Path::new("link");
    ///     symlink("/origin_does_not_exist/", link_path)?;
    ///
    ///     let metadata = fs::symlink_metadata(link_path)?;
    ///
    ///     assert!(metadata.is_symlink());
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "is_symlink", since = "1.58.0")]
    pub fn is_symlink(&self) -> bool {
        self.file_type().is_symlink()
    }

    /// Returns the size of the file, in bytes, this metadata is for.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let metadata = fs::metadata("foo.txt")?;
    ///
    ///     assert_eq!(0, metadata.len());
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> u64 {
        self.0.size()
    }

    /// Returns the permissions of the file this metadata is for.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let metadata = fs::metadata("foo.txt")?;
    ///
    ///     assert!(!metadata.permissions().readonly());
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn permissions(&self) -> Permissions {
        Permissions(self.0.perm())
    }

    /// Returns the last modification time listed in this metadata.
    ///
    /// The returned value corresponds to the `mtime` field of `stat` on Unix
    /// platforms and the `ftLastWriteTime` field on Windows platforms.
    ///
    /// # Errors
    ///
    /// This field might not be available on all platforms, and will return an
    /// `Err` on platforms where it is not available.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let metadata = fs::metadata("foo.txt")?;
    ///
    ///     if let Ok(time) = metadata.modified() {
    ///         println!("{time:?}");
    ///     } else {
    ///         println!("Not supported on this platform");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[doc(alias = "mtime", alias = "ftLastWriteTime")]
    #[stable(feature = "fs_time", since = "1.10.0")]
    pub fn modified(&self) -> io::Result<SystemTime> {
        self.0.modified().map(FromInner::from_inner)
    }

    /// Returns the last access time of this metadata.
    ///
    /// The returned value corresponds to the `atime` field of `stat` on Unix
    /// platforms and the `ftLastAccessTime` field on Windows platforms.
    ///
    /// Note that not all platforms will keep this field update in a file's
    /// metadata, for example Windows has an option to disable updating this
    /// time when files are accessed and Linux similarly has `noatime`.
    ///
    /// # Errors
    ///
    /// This field might not be available on all platforms, and will return an
    /// `Err` on platforms where it is not available.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let metadata = fs::metadata("foo.txt")?;
    ///
    ///     if let Ok(time) = metadata.accessed() {
    ///         println!("{time:?}");
    ///     } else {
    ///         println!("Not supported on this platform");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[doc(alias = "atime", alias = "ftLastAccessTime")]
    #[stable(feature = "fs_time", since = "1.10.0")]
    pub fn accessed(&self) -> io::Result<SystemTime> {
        self.0.accessed().map(FromInner::from_inner)
    }

    /// Returns the creation time listed in this metadata.
    ///
    /// The returned value corresponds to the `btime` field of `statx` on
    /// Linux kernel starting from to 4.11, the `birthtime` field of `stat` on other
    /// Unix platforms, and the `ftCreationTime` field on Windows platforms.
    ///
    /// # Errors
    ///
    /// This field might not be available on all platforms, and will return an
    /// `Err` on platforms or filesystems where it is not available.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let metadata = fs::metadata("foo.txt")?;
    ///
    ///     if let Ok(time) = metadata.created() {
    ///         println!("{time:?}");
    ///     } else {
    ///         println!("Not supported on this platform or filesystem");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[doc(alias = "btime", alias = "birthtime", alias = "ftCreationTime")]
    #[stable(feature = "fs_time", since = "1.10.0")]
    pub fn created(&self) -> io::Result<SystemTime> {
        self.0.created().map(FromInner::from_inner)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("Metadata");
        debug.field("file_type", &self.file_type());
        debug.field("permissions", &self.permissions());
        debug.field("len", &self.len());
        if let Ok(modified) = self.modified() {
            debug.field("modified", &modified);
        }
        if let Ok(accessed) = self.accessed() {
            debug.field("accessed", &accessed);
        }
        if let Ok(created) = self.created() {
            debug.field("created", &created);
        }
        debug.finish_non_exhaustive()
    }
}

impl AsInner<fs_imp::FileAttr> for Metadata {
    #[inline]
    fn as_inner(&self) -> &fs_imp::FileAttr {
        &self.0
    }
}

impl FromInner<fs_imp::FileAttr> for Metadata {
    fn from_inner(attr: fs_imp::FileAttr) -> Metadata {
        Metadata(attr)
    }
}

impl FileTimes {
    /// Creates a new `FileTimes` with no times set.
    ///
    /// Using the resulting `FileTimes` in [`File::set_times`] will not modify any timestamps.
    #[stable(feature = "file_set_times", since = "1.75.0")]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the last access time of a file.
    #[stable(feature = "file_set_times", since = "1.75.0")]
    pub fn set_accessed(mut self, t: SystemTime) -> Self {
        self.0.set_accessed(t.into_inner());
        self
    }

    /// Set the last modified time of a file.
    #[stable(feature = "file_set_times", since = "1.75.0")]
    pub fn set_modified(mut self, t: SystemTime) -> Self {
        self.0.set_modified(t.into_inner());
        self
    }
}

impl AsInnerMut<fs_imp::FileTimes> for FileTimes {
    fn as_inner_mut(&mut self) -> &mut fs_imp::FileTimes {
        &mut self.0
    }
}

// For implementing OS extension traits in `std::os`
#[stable(feature = "file_set_times", since = "1.75.0")]
impl Sealed for FileTimes {}

impl Permissions {
    /// Returns `true` if these permissions describe a readonly (unwritable) file.
    ///
    /// # Note
    ///
    /// This function does not take Access Control Lists (ACLs), Unix group
    /// membership and other nuances into account.
    /// Therefore the return value of this function cannot be relied upon
    /// to predict whether attempts to read or write the file will actually succeed.
    ///
    /// # Windows
    ///
    /// On Windows this returns [`FILE_ATTRIBUTE_READONLY`](https://docs.microsoft.com/en-us/windows/win32/fileio/file-attribute-constants).
    /// If `FILE_ATTRIBUTE_READONLY` is set then writes to the file will fail
    /// but the user may still have permission to change this flag. If
    /// `FILE_ATTRIBUTE_READONLY` is *not* set then writes may still fail due
    /// to lack of write permission.
    /// The behavior of this attribute for directories depends on the Windows
    /// version.
    ///
    /// # Unix (including macOS)
    ///
    /// On Unix-based platforms this checks if *any* of the owner, group or others
    /// write permission bits are set. It does not consider anything else, including:
    ///
    /// * Whether the current user is in the file's assigned group.
    /// * Permissions granted by ACL.
    /// * That `root` user can write to files that do not have any write bits set.
    /// * Writable files on a filesystem that is mounted read-only.
    ///
    /// The [`PermissionsExt`] trait gives direct access to the permission bits but
    /// also does not read ACLs.
    ///
    /// [`PermissionsExt`]: crate::os::unix::fs::PermissionsExt
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::create("foo.txt")?;
    ///     let metadata = f.metadata()?;
    ///
    ///     assert_eq!(false, metadata.permissions().readonly());
    ///     Ok(())
    /// }
    /// ```
    #[must_use = "call `set_readonly` to modify the readonly flag"]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn readonly(&self) -> bool {
        self.0.readonly()
    }

    /// Modifies the readonly flag for this set of permissions. If the
    /// `readonly` argument is `true`, using the resulting `Permission` will
    /// update file permissions to forbid writing. Conversely, if it's `false`,
    /// using the resulting `Permission` will update file permissions to allow
    /// writing.
    ///
    /// This operation does **not** modify the files attributes. This only
    /// changes the in-memory value of these attributes for this `Permissions`
    /// instance. To modify the files attributes use the [`set_permissions`]
    /// function which commits these attribute changes to the file.
    ///
    /// # Note
    ///
    /// `set_readonly(false)` makes the file *world-writable* on Unix.
    /// You can use the [`PermissionsExt`] trait on Unix to avoid this issue.
    ///
    /// It also does not take Access Control Lists (ACLs) or Unix group
    /// membership into account.
    ///
    /// # Windows
    ///
    /// On Windows this sets or clears [`FILE_ATTRIBUTE_READONLY`](https://docs.microsoft.com/en-us/windows/win32/fileio/file-attribute-constants).
    /// If `FILE_ATTRIBUTE_READONLY` is set then writes to the file will fail
    /// but the user may still have permission to change this flag. If
    /// `FILE_ATTRIBUTE_READONLY` is *not* set then the write may still fail if
    /// the user does not have permission to write to the file.
    ///
    /// In Windows 7 and earlier this attribute prevents deleting empty
    /// directories. It does not prevent modifying the directory contents.
    /// On later versions of Windows this attribute is ignored for directories.
    ///
    /// # Unix (including macOS)
    ///
    /// On Unix-based platforms this sets or clears the write access bit for
    /// the owner, group *and* others, equivalent to `chmod a+w <file>`
    /// or `chmod a-w <file>` respectively. The latter will grant write access
    /// to all users! You can use the [`PermissionsExt`] trait on Unix
    /// to avoid this issue.
    ///
    /// [`PermissionsExt`]: crate::os::unix::fs::PermissionsExt
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::create("foo.txt")?;
    ///     let metadata = f.metadata()?;
    ///     let mut permissions = metadata.permissions();
    ///
    ///     permissions.set_readonly(true);
    ///
    ///     // filesystem doesn't change, only the in memory state of the
    ///     // readonly permission
    ///     assert_eq!(false, metadata.permissions().readonly());
    ///
    ///     // just this particular `permissions`.
    ///     assert_eq!(true, permissions.readonly());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set_readonly(&mut self, readonly: bool) {
        self.0.set_readonly(readonly)
    }
}

impl FileType {
    /// Tests whether this file type represents a directory. The
    /// result is mutually exclusive to the results of
    /// [`is_file`] and [`is_symlink`]; only zero or one of these
    /// tests may pass.
    ///
    /// [`is_file`]: FileType::is_file
    /// [`is_symlink`]: FileType::is_symlink
    ///
    /// # Examples
    ///
    /// ```no_run
    /// fn main() -> std::io::Result<()> {
    ///     use std::fs;
    ///
    ///     let metadata = fs::metadata("foo.txt")?;
    ///     let file_type = metadata.file_type();
    ///
    ///     assert_eq!(file_type.is_dir(), false);
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn is_dir(&self) -> bool {
        self.0.is_dir()
    }

    /// Tests whether this file type represents a regular file.
    /// The result is mutually exclusive to the results of
    /// [`is_dir`] and [`is_symlink`]; only zero or one of these
    /// tests may pass.
    ///
    /// When the goal is simply to read from (or write to) the source, the most
    /// reliable way to test the source can be read (or written to) is to open
    /// it. Only using `is_file` can break workflows like `diff <( prog_a )` on
    /// a Unix-like system for example. See [`File::open`] or
    /// [`OpenOptions::open`] for more information.
    ///
    /// [`is_dir`]: FileType::is_dir
    /// [`is_symlink`]: FileType::is_symlink
    ///
    /// # Examples
    ///
    /// ```no_run
    /// fn main() -> std::io::Result<()> {
    ///     use std::fs;
    ///
    ///     let metadata = fs::metadata("foo.txt")?;
    ///     let file_type = metadata.file_type();
    ///
    ///     assert_eq!(file_type.is_file(), true);
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn is_file(&self) -> bool {
        self.0.is_file()
    }

    /// Tests whether this file type represents a symbolic link.
    /// The result is mutually exclusive to the results of
    /// [`is_dir`] and [`is_file`]; only zero or one of these
    /// tests may pass.
    ///
    /// The underlying [`Metadata`] struct needs to be retrieved
    /// with the [`fs::symlink_metadata`] function and not the
    /// [`fs::metadata`] function. The [`fs::metadata`] function
    /// follows symbolic links, so [`is_symlink`] would always
    /// return `false` for the target file.
    ///
    /// [`fs::metadata`]: metadata
    /// [`fs::symlink_metadata`]: symlink_metadata
    /// [`is_dir`]: FileType::is_dir
    /// [`is_file`]: FileType::is_file
    /// [`is_symlink`]: FileType::is_symlink
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let metadata = fs::symlink_metadata("foo.txt")?;
    ///     let file_type = metadata.file_type();
    ///
    ///     assert_eq!(file_type.is_symlink(), false);
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn is_symlink(&self) -> bool {
        self.0.is_symlink()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for FileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileType")
            .field("is_file", &self.is_file())
            .field("is_dir", &self.is_dir())
            .field("is_symlink", &self.is_symlink())
            .finish_non_exhaustive()
    }
}

impl AsInner<fs_imp::FileType> for FileType {
    #[inline]
    fn as_inner(&self) -> &fs_imp::FileType {
        &self.0
    }
}

impl FromInner<fs_imp::FilePermissions> for Permissions {
    fn from_inner(f: fs_imp::FilePermissions) -> Permissions {
        Permissions(f)
    }
}

impl AsInner<fs_imp::FilePermissions> for Permissions {
    #[inline]
    fn as_inner(&self) -> &fs_imp::FilePermissions {
        &self.0
    }
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
    /// The full path is created by joining the original path to `read_dir`
    /// with the filename of this entry.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     for entry in fs::read_dir(".")? {
    ///         let dir = entry?;
    ///         println!("{:?}", dir.path());
    ///     }
    ///     Ok(())
    /// }
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn path(&self) -> PathBuf {
        self.0.path()
    }

    /// Returns the metadata for the file that this entry points at.
    ///
    /// This function will not traverse symlinks if this entry points at a
    /// symlink. To traverse symlinks use [`fs::metadata`] or [`fs::File::metadata`].
    ///
    /// [`fs::metadata`]: metadata
    /// [`fs::File::metadata`]: File::metadata
    ///
    /// # Platform-specific behavior
    ///
    /// On Windows this function is cheap to call (no extra system calls
    /// needed), but on Unix platforms this function is the equivalent of
    /// calling `symlink_metadata` on the path.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs;
    ///
    /// if let Ok(entries) = fs::read_dir(".") {
    ///     for entry in entries {
    ///         if let Ok(entry) = entry {
    ///             // Here, `entry` is a `DirEntry`.
    ///             if let Ok(metadata) = entry.metadata() {
    ///                 // Now let's show our entry's permissions!
    ///                 println!("{:?}: {:?}", entry.path(), metadata.permissions());
    ///             } else {
    ///                 println!("Couldn't get metadata for {:?}", entry.path());
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    #[stable(feature = "dir_entry_ext", since = "1.1.0")]
    pub fn metadata(&self) -> io::Result<Metadata> {
        self.0.metadata().map(Metadata)
    }

    /// Returns the file type for the file that this entry points at.
    ///
    /// This function will not traverse symlinks if this entry points at a
    /// symlink.
    ///
    /// # Platform-specific behavior
    ///
    /// On Windows and most Unix platforms this function is free (no extra
    /// system calls needed), but some Unix platforms may require the equivalent
    /// call to `symlink_metadata` to learn about the target file type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs;
    ///
    /// if let Ok(entries) = fs::read_dir(".") {
    ///     for entry in entries {
    ///         if let Ok(entry) = entry {
    ///             // Here, `entry` is a `DirEntry`.
    ///             if let Ok(file_type) = entry.file_type() {
    ///                 // Now let's show our entry's file type!
    ///                 println!("{:?}: {:?}", entry.path(), file_type);
    ///             } else {
    ///                 println!("Couldn't get file type for {:?}", entry.path());
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    #[stable(feature = "dir_entry_ext", since = "1.1.0")]
    pub fn file_type(&self) -> io::Result<FileType> {
        self.0.file_type().map(FileType)
    }

    /// Returns the file name of this directory entry without any
    /// leading path component(s).
    ///
    /// As an example,
    /// the output of the function will result in "foo" for all the following paths:
    /// - "./foo"
    /// - "/the/foo"
    /// - "../../foo"
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs;
    ///
    /// if let Ok(entries) = fs::read_dir(".") {
    ///     for entry in entries {
    ///         if let Ok(entry) = entry {
    ///             // Here, `entry` is a `DirEntry`.
    ///             println!("{:?}", entry.file_name());
    ///         }
    ///     }
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "dir_entry_ext", since = "1.1.0")]
    pub fn file_name(&self) -> OsString {
        self.0.file_name()
    }
}

#[stable(feature = "dir_entry_debug", since = "1.13.0")]
impl fmt::Debug for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DirEntry").field(&self.path()).finish()
    }
}

impl AsInner<fs_imp::DirEntry> for DirEntry {
    #[inline]
    fn as_inner(&self) -> &fs_imp::DirEntry {
        &self.0
    }
}

/// Removes a file from the filesystem.
///
/// Note that there is no
/// guarantee that the file is immediately deleted (e.g., depending on
/// platform, other open file descriptors may prevent immediate removal).
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `unlink` function on Unix.
/// On Windows, `DeleteFile` is used or `CreateFileW` and `SetInformationByHandle` for readonly files.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * `path` points to a directory.
/// * The file doesn't exist.
/// * The user lacks permissions to remove the file.
///
/// This function will only ever return an error of kind `NotFound` if the given
/// path does not exist. Note that the inverse is not true,
/// ie. if a path does not exist, its removal may fail for a number of reasons,
/// such as insufficient permissions.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::remove_file("a.txt")?;
///     Ok(())
/// }
/// ```
#[doc(alias = "rm", alias = "unlink", alias = "DeleteFile")]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()> {
    fs_imp::unlink(path.as_ref())
}

/// Given a path, queries the file system to get information about a file,
/// directory, etc.
///
/// This function will traverse symbolic links to query information about the
/// destination file.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `stat` function on Unix
/// and the `GetFileInformationByHandle` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The user lacks permissions to perform `metadata` call on `path`.
/// * `path` does not exist.
///
/// # Examples
///
/// ```rust,no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     let attr = fs::metadata("/some/file/path.txt")?;
///     // inspect attr ...
///     Ok(())
/// }
/// ```
#[doc(alias = "stat")]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn metadata<P: AsRef<Path>>(path: P) -> io::Result<Metadata> {
    fs_imp::stat(path.as_ref()).map(Metadata)
}

/// Queries the metadata about a file without following symlinks.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `lstat` function on Unix
/// and the `GetFileInformationByHandle` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The user lacks permissions to perform `metadata` call on `path`.
/// * `path` does not exist.
///
/// # Examples
///
/// ```rust,no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     let attr = fs::symlink_metadata("/some/file/path.txt")?;
///     // inspect attr ...
///     Ok(())
/// }
/// ```
#[doc(alias = "lstat")]
#[stable(feature = "symlink_metadata", since = "1.1.0")]
pub fn symlink_metadata<P: AsRef<Path>>(path: P) -> io::Result<Metadata> {
    fs_imp::lstat(path.as_ref()).map(Metadata)
}

/// Renames a file or directory to a new name, replacing the original file if
/// `to` already exists.
///
/// This will not work if the new name is on a different mount point.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `rename` function on Unix
/// and the `SetFileInformationByHandle` function on Windows.
///
/// Because of this, the behavior when both `from` and `to` exist differs. On
/// Unix, if `from` is a directory, `to` must also be an (empty) directory. If
/// `from` is not a directory, `to` must also be not a directory. The behavior
/// on Windows is the same on Windows 10 1607 and higher if `FileRenameInfoEx`
/// is supported by the filesystem; otherwise, `from` can be anything, but
/// `to` must *not* be a directory.
///
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * `from` does not exist.
/// * The user lacks permissions to view contents.
/// * `from` and `to` are on separate filesystems.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::rename("a.txt", "b.txt")?; // Rename a.txt to b.txt
///     Ok(())
/// }
/// ```
#[doc(alias = "mv", alias = "MoveFile", alias = "MoveFileEx")]
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
/// On success, the total number of bytes copied is returned and it is equal to
/// the length of the `to` file as reported by `metadata`.
///
/// If you want to copy the contents of one file to another and you’re
/// working with [`File`]s, see the [`io::copy`](io::copy()) function.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `open` function in Unix
/// with `O_RDONLY` for `from` and `O_WRONLY`, `O_CREAT`, and `O_TRUNC` for `to`.
/// `O_CLOEXEC` is set for returned file descriptors.
///
/// On Linux (including Android), this function attempts to use `copy_file_range(2)`,
/// and falls back to reading and writing if that is not possible.
///
/// On Windows, this function currently corresponds to `CopyFileEx`. Alternate
/// NTFS streams are copied but only the size of the main stream is returned by
/// this function.
///
/// On MacOS, this function corresponds to `fclonefileat` and `fcopyfile`.
///
/// Note that platform-specific behavior [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * `from` is neither a regular file nor a symlink to a regular file.
/// * `from` does not exist.
/// * The current process does not have the permission rights to read
///   `from` or write `to`.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::copy("foo.txt", "bar.txt")?;  // Copy foo.txt to bar.txt
///     Ok(())
/// }
/// ```
#[doc(alias = "cp")]
#[doc(alias = "CopyFile", alias = "CopyFileEx")]
#[doc(alias = "fclonefileat", alias = "fcopyfile")]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
    fs_imp::copy(from.as_ref(), to.as_ref())
}

/// Creates a new hard link on the filesystem.
///
/// The `link` path will be a link pointing to the `original` path. Note that
/// systems often require these two paths to both be located on the same
/// filesystem.
///
/// If `original` names a symbolic link, it is platform-specific whether the
/// symbolic link is followed. On platforms where it's possible to not follow
/// it, it is not followed, and the created hard link points to the symbolic
/// link itself.
///
/// # Platform-specific behavior
///
/// This function currently corresponds the `CreateHardLink` function on Windows.
/// On most Unix systems, it corresponds to the `linkat` function with no flags.
/// On Android, VxWorks, and Redox, it instead corresponds to the `link` function.
/// On MacOS, it uses the `linkat` function if it is available, but on very old
/// systems where `linkat` is not available, `link` is selected at runtime instead.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The `original` path is not a file or doesn't exist.
/// * The 'link' path already exists.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::hard_link("a.txt", "b.txt")?; // Hard link a.txt to b.txt
///     Ok(())
/// }
/// ```
#[doc(alias = "CreateHardLink", alias = "linkat")]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn hard_link<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) -> io::Result<()> {
    fs_imp::link(original.as_ref(), link.as_ref())
}

/// Creates a new symbolic link on the filesystem.
///
/// The `link` path will be a symbolic link pointing to the `original` path.
/// On Windows, this will be a file symlink, not a directory symlink;
/// for this reason, the platform-specific [`std::os::unix::fs::symlink`]
/// and [`std::os::windows::fs::symlink_file`] or [`symlink_dir`] should be
/// used instead to make the intent explicit.
///
/// [`std::os::unix::fs::symlink`]: crate::os::unix::fs::symlink
/// [`std::os::windows::fs::symlink_file`]: crate::os::windows::fs::symlink_file
/// [`symlink_dir`]: crate::os::windows::fs::symlink_dir
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::soft_link("a.txt", "b.txt")?;
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(
    since = "1.1.0",
    note = "replaced with std::os::unix::fs::symlink and \
            std::os::windows::fs::{symlink_file, symlink_dir}"
)]
pub fn soft_link<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) -> io::Result<()> {
    fs_imp::symlink(original.as_ref(), link.as_ref())
}

/// Reads a symbolic link, returning the file that the link points to.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `readlink` function on Unix
/// and the `CreateFile` function with `FILE_FLAG_OPEN_REPARSE_POINT` and
/// `FILE_FLAG_BACKUP_SEMANTICS` flags on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * `path` is not a symbolic link.
/// * `path` does not exist.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     let path = fs::read_link("a.txt")?;
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn read_link<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    fs_imp::readlink(path.as_ref())
}

/// Returns the canonical, absolute form of a path with all intermediate
/// components normalized and symbolic links resolved.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `realpath` function on Unix
/// and the `CreateFile` and `GetFinalPathNameByHandle` functions on Windows.
/// Note that this [may change in the future][changes].
///
/// On Windows, this converts the path to use [extended length path][path]
/// syntax, which allows your program to use longer path names, but means you
/// can only join backslash-delimited paths to it, and it may be incompatible
/// with other applications (if passed to the application on the command-line,
/// or written to a file another application may read).
///
/// [changes]: io#platform-specific-behavior
/// [path]: https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * `path` does not exist.
/// * A non-final component in path is not a directory.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     let path = fs::canonicalize("../a/../foo.txt")?;
///     Ok(())
/// }
/// ```
#[doc(alias = "realpath")]
#[doc(alias = "GetFinalPathNameByHandle")]
#[stable(feature = "fs_canonicalize", since = "1.5.0")]
pub fn canonicalize<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    fs_imp::canonicalize(path.as_ref())
}

/// Creates a new, empty directory at the provided path
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `mkdir` function on Unix
/// and the `CreateDirectoryW` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// **NOTE**: If a parent of the given path doesn't exist, this function will
/// return an error. To create a directory and all its missing parents at the
/// same time, use the [`create_dir_all`] function.
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * User lacks permissions to create directory at `path`.
/// * A parent of the given path doesn't exist. (To create a directory and all
///   its missing parents at the same time, use the [`create_dir_all`]
///   function.)
/// * `path` already exists.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::create_dir("/some/dir")?;
///     Ok(())
/// }
/// ```
#[doc(alias = "mkdir", alias = "CreateDirectory")]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "fs_create_dir")]
pub fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    DirBuilder::new().create(path.as_ref())
}

/// Recursively create a directory and all of its parent components if they
/// are missing.
///
/// If this function returns an error, some of the parent components might have
/// been created already.
///
/// If the empty path is passed to this function, it always succeeds without
/// creating any directories.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to multiple calls to the `mkdir`
/// function on Unix and the `CreateDirectoryW` function on Windows.
///
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// The function will return an error if any directory specified in path does not exist and
/// could not be created. There may be other error conditions; see [`fs::create_dir`] for specifics.
///
/// Notable exception is made for situations where any of the directories
/// specified in the `path` could not be created as it was being created concurrently.
/// Such cases are considered to be successful. That is, calling `create_dir_all`
/// concurrently from multiple threads or processes is guaranteed not to fail
/// due to a race condition with itself.
///
/// [`fs::create_dir`]: create_dir
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::create_dir_all("/some/dir")?;
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    DirBuilder::new().recursive(true).create(path.as_ref())
}

/// Removes an empty directory.
///
/// If you want to remove a directory that is not empty, as well as all
/// of its contents recursively, consider using [`remove_dir_all`]
/// instead.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `rmdir` function on Unix
/// and the `RemoveDirectory` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * `path` doesn't exist.
/// * `path` isn't a directory.
/// * The user lacks permissions to remove the directory at the provided `path`.
/// * The directory isn't empty.
///
/// This function will only ever return an error of kind `NotFound` if the given
/// path does not exist. Note that the inverse is not true,
/// ie. if a path does not exist, its removal may fail for a number of reasons,
/// such as insufficient permissions.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::remove_dir("/some/dir")?;
///     Ok(())
/// }
/// ```
#[doc(alias = "rmdir", alias = "RemoveDirectory")]
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
/// # Platform-specific behavior
///
/// This function currently corresponds to `openat`, `fdopendir`, `unlinkat` and `lstat` functions
/// on Unix (except for REDOX) and the `CreateFileW`, `GetFileInformationByHandleEx`,
/// `SetFileInformationByHandle`, and `NtCreateFile` functions on Windows. Note that, this
/// [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// On REDOX, as well as when running in Miri for any target, this function is not protected against
/// time-of-check to time-of-use (TOCTOU) race conditions, and should not be used in
/// security-sensitive code on those platforms. All other platforms are protected.
///
/// # Errors
///
/// See [`fs::remove_file`] and [`fs::remove_dir`].
///
/// `remove_dir_all` will fail if `remove_dir` or `remove_file` fail on any constituent paths, including the root `path`.
/// As a result, the directory you are deleting must exist, meaning that this function is not idempotent.
/// Additionally, `remove_dir_all` will also fail if the `path` is not a directory.
///
/// Consider ignoring the error if validating the removal is not required for your use case.
///
/// [`io::ErrorKind::NotFound`] is only returned if no removal occurs.
///
/// [`fs::remove_file`]: remove_file
/// [`fs::remove_dir`]: remove_dir
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     fs::remove_dir_all("/some/dir")?;
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    fs_imp::remove_dir_all(path.as_ref())
}

/// Returns an iterator over the entries within a directory.
///
/// The iterator will yield instances of <code>[io::Result]<[DirEntry]></code>.
/// New errors may be encountered after an iterator is initially constructed.
/// Entries for the current and parent directories (typically `.` and `..`) are
/// skipped.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `opendir` function on Unix
/// and the `FindFirstFileEx` function on Windows. Advancing the iterator
/// currently corresponds to `readdir` on Unix and `FindNextFile` on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// The order in which this iterator returns entries is platform and filesystem
/// dependent.
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The provided `path` doesn't exist.
/// * The process lacks permissions to view the contents.
/// * The `path` points at a non-directory file.
///
/// # Examples
///
/// ```
/// use std::io;
/// use std::fs::{self, DirEntry};
/// use std::path::Path;
///
/// // one possible implementation of walking a directory only visiting files
/// fn visit_dirs(dir: &Path, cb: &dyn Fn(&DirEntry)) -> io::Result<()> {
///     if dir.is_dir() {
///         for entry in fs::read_dir(dir)? {
///             let entry = entry?;
///             let path = entry.path();
///             if path.is_dir() {
///                 visit_dirs(&path, cb)?;
///             } else {
///                 cb(&entry);
///             }
///         }
///     }
///     Ok(())
/// }
/// ```
///
/// ```rust,no_run
/// use std::{fs, io};
///
/// fn main() -> io::Result<()> {
///     let mut entries = fs::read_dir(".")?
///         .map(|res| res.map(|e| e.path()))
///         .collect::<Result<Vec<_>, io::Error>>()?;
///
///     // The order in which `read_dir` returns entries is not guaranteed. If reproducible
///     // ordering is required the entries should be explicitly sorted.
///
///     entries.sort();
///
///     // The entries have now been sorted by their path.
///
///     Ok(())
/// }
/// ```
#[doc(alias = "ls", alias = "opendir", alias = "FindFirstFile", alias = "FindNextFile")]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn read_dir<P: AsRef<Path>>(path: P) -> io::Result<ReadDir> {
    fs_imp::readdir(path.as_ref()).map(ReadDir)
}

/// Changes the permissions found on a file or a directory.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `chmod` function on Unix
/// and the `SetFileAttributes` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * `path` does not exist.
/// * The user lacks the permission to change attributes of the file.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// fn main() -> std::io::Result<()> {
///     let mut perms = fs::metadata("foo.txt")?.permissions();
///     perms.set_readonly(true);
///     fs::set_permissions("foo.txt", perms)?;
///     Ok(())
/// }
/// ```
#[doc(alias = "chmod", alias = "SetFileAttributes")]
#[stable(feature = "set_permissions", since = "1.1.0")]
pub fn set_permissions<P: AsRef<Path>>(path: P, perm: Permissions) -> io::Result<()> {
    fs_imp::set_perm(path.as_ref(), perm.0)
}

impl DirBuilder {
    /// Creates a new set of options with default mode/security settings for all
    /// platforms and also non-recursive.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::DirBuilder;
    ///
    /// let builder = DirBuilder::new();
    /// ```
    #[stable(feature = "dir_builder", since = "1.6.0")]
    #[must_use]
    pub fn new() -> DirBuilder {
        DirBuilder { inner: fs_imp::DirBuilder::new(), recursive: false }
    }

    /// Indicates that directories should be created recursively, creating all
    /// parent directories. Parents that do not exist are created with the same
    /// security and permissions settings.
    ///
    /// This option defaults to `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::DirBuilder;
    ///
    /// let mut builder = DirBuilder::new();
    /// builder.recursive(true);
    /// ```
    #[stable(feature = "dir_builder", since = "1.6.0")]
    pub fn recursive(&mut self, recursive: bool) -> &mut Self {
        self.recursive = recursive;
        self
    }

    /// Creates the specified directory with the options configured in this
    /// builder.
    ///
    /// It is considered an error if the directory already exists unless
    /// recursive mode is enabled.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::{self, DirBuilder};
    ///
    /// let path = "/tmp/foo/bar/baz";
    /// DirBuilder::new()
    ///     .recursive(true)
    ///     .create(path).unwrap();
    ///
    /// assert!(fs::metadata(path).unwrap().is_dir());
    /// ```
    #[stable(feature = "dir_builder", since = "1.6.0")]
    pub fn create<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        self._create(path.as_ref())
    }

    fn _create(&self, path: &Path) -> io::Result<()> {
        if self.recursive { self.create_dir_all(path) } else { self.inner.mkdir(path) }
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        if path == Path::new("") {
            return Ok(());
        }

        match self.inner.mkdir(path) {
            Ok(()) => return Ok(()),
            Err(ref e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(_) if path.is_dir() => return Ok(()),
            Err(e) => return Err(e),
        }
        match path.parent() {
            Some(p) => self.create_dir_all(p)?,
            None => {
                return Err(io::const_error!(
                    io::ErrorKind::Uncategorized,
                    "failed to create whole tree",
                ));
            }
        }
        match self.inner.mkdir(path) {
            Ok(()) => Ok(()),
            Err(_) if path.is_dir() => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl AsInnerMut<fs_imp::DirBuilder> for DirBuilder {
    #[inline]
    fn as_inner_mut(&mut self) -> &mut fs_imp::DirBuilder {
        &mut self.inner
    }
}

/// Returns `Ok(true)` if the path points at an existing entity.
///
/// This function will traverse symbolic links to query information about the
/// destination file. In case of broken symbolic links this will return `Ok(false)`.
///
/// As opposed to the [`Path::exists`] method, this will only return `Ok(true)` or `Ok(false)`
/// if the path was _verified_ to exist or not exist. If its existence can neither be confirmed
/// nor denied, an `Err(_)` will be propagated instead. This can be the case if e.g. listing
/// permission is denied on one of the parent directories.
///
/// Note that while this avoids some pitfalls of the `exists()` method, it still can not
/// prevent time-of-check to time-of-use (TOCTOU) bugs. You should only use it in scenarios
/// where those bugs are not an issue.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
///
/// assert!(!fs::exists("does_not_exist.txt").expect("Can't check existence of file does_not_exist.txt"));
/// assert!(fs::exists("/root/secret_file.txt").is_err());
/// ```
///
/// [`Path::exists`]: crate::path::Path::exists
#[stable(feature = "fs_try_exists", since = "1.81.0")]
#[inline]
pub fn exists<P: AsRef<Path>>(path: P) -> io::Result<bool> {
    fs_imp::exists(path.as_ref())
}
