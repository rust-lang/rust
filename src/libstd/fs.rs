// ignore-tidy-filelength

//! Filesystem manipulation operations.
//!
//! This module contains basic methods to manipulate the contents of the local
//! filesystem. All methods in this module represent cross-platform filesystem
//! operations. Extra platform-specific functionality can be found in the
//! extension traits of `std::os::$platform`.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::fmt;
use crate::ffi::OsString;
use crate::io::{self, SeekFrom, Seek, Read, Initializer, Write, IoSlice, IoSliceMut};
use crate::path::{Path, PathBuf};
use crate::sys::fs as fs_imp;
use crate::sys_common::{AsInnerMut, FromInner, AsInner, IntoInner};
use crate::time::SystemTime;

/// A reference to an open file on the filesystem.
///
/// An instance of a `File` can be read and/or written depending on what options
/// it was opened with. Files also implement [`Seek`] to alter the logical cursor
/// that the file contains internally.
///
/// Files are automatically closed when they go out of scope.  Errors detected
/// on closing are ignored by the implementation of `Drop`.  Use the method
/// [`sync_all`] if these errors must be manually handled.
///
/// # Examples
///
/// Creates a new file and write bytes to it:
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
/// Read the contents of a file into a [`String`]:
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
/// It can be more efficient to read the contents of a file with a buffered
/// [`Read`]er. This can be accomplished with [`BufReader<R>`]:
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
/// [`Seek`]: ../io/trait.Seek.html
/// [`String`]: ../string/struct.String.html
/// [`Read`]: ../io/trait.Read.html
/// [`Write`]: ../io/trait.Write.html
/// [`BufReader<R>`]: ../io/struct.BufReader.html
/// [`sync_all`]: struct.File.html#method.sync_all
#[stable(feature = "rust1", since = "1.0.0")]
pub struct File {
    inner: fs_imp::File,
}

/// Metadata information about a file.
///
/// This structure is returned from the [`metadata`] or
/// [`symlink_metadata`] function or method and represents known
/// metadata about a file such as its permissions, size, modification
/// times, etc.
///
/// [`metadata`]: fn.metadata.html
/// [`symlink_metadata`]: fn.symlink_metadata.html
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Metadata(fs_imp::FileAttr);

/// Iterator over the entries in a directory.
///
/// This iterator is returned from the [`read_dir`] function of this module and
/// will yield instances of [`io::Result`]`<`[`DirEntry`]`>`. Through a [`DirEntry`]
/// information like the entry's path and possibly other metadata can be
/// learned.
///
/// # Errors
///
/// This [`io::Result`] will be an [`Err`] if there's some sort of intermittent
/// IO error during iteration.
///
/// [`read_dir`]: fn.read_dir.html
/// [`DirEntry`]: struct.DirEntry.html
/// [`io::Result`]: ../io/type.Result.html
/// [`Err`]: ../result/enum.Result.html#variant.Err
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct ReadDir(fs_imp::ReadDir);

/// Entries returned by the [`ReadDir`] iterator.
///
/// [`ReadDir`]: struct.ReadDir.html
///
/// An instance of `DirEntry` represents an entry inside of a directory on the
/// filesystem. Each entry can be inspected via methods to learn about the full
/// path or possibly other metadata through per-platform extension traits.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct DirEntry(fs_imp::DirEntry);

/// Options and flags which can be used to configure how a file is opened.
///
/// This builder exposes the ability to configure how a [`File`] is opened and
/// what operations are permitted on the open file. The [`File::open`] and
/// [`File::create`] methods are aliases for commonly used options using this
/// builder.
///
/// [`File`]: struct.File.html
/// [`File::open`]: struct.File.html#method.open
/// [`File::create`]: struct.File.html#method.create
///
/// Generally speaking, when using `OpenOptions`, you'll first call [`new`],
/// then chain calls to methods to set each option, then call [`open`],
/// passing the path of the file you're trying to open. This will give you a
/// [`io::Result`][result] with a [`File`][file] inside that you can further
/// operate on.
///
/// [`new`]: struct.OpenOptions.html#method.new
/// [`open`]: struct.OpenOptions.html#method.open
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
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OpenOptions(fs_imp::OpenOptions);

/// Representation of the various permissions on a file.
///
/// This module only currently provides one bit of information, [`readonly`],
/// which is exposed on all currently supported platforms. Unix-specific
/// functionality, such as mode bits, is available through the
/// [`PermissionsExt`] trait.
///
/// [`readonly`]: struct.Permissions.html#method.readonly
/// [`PermissionsExt`]: ../os/unix/fs/trait.PermissionsExt.html
#[derive(Clone, PartialEq, Eq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Permissions(fs_imp::FilePermissions);

/// A structure representing a type of file with accessors for each file type.
/// It is returned by [`Metadata::file_type`] method.
///
/// [`Metadata::file_type`]: struct.Metadata.html#method.file_type
#[stable(feature = "file_type", since = "1.1.0")]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileType(fs_imp::FileType);

/// A builder used to create directories in various manners.
///
/// This builder also supports platform-specific options.
#[stable(feature = "dir_builder", since = "1.6.0")]
#[derive(Debug)]
pub struct DirBuilder {
    inner: fs_imp::DirBuilder,
    recursive: bool,
}

/// Indicates how large a buffer to pre-allocate before reading the entire file.
fn initial_buffer_size(file: &File) -> usize {
    // Allocate one extra byte so the buffer doesn't need to grow before the
    // final `read` call at the end of the file.  Don't worry about `usize`
    // overflow because reading will fail regardless in that case.
    file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0)
}

/// Read the entire contents of a file into a bytes vector.
///
/// This is a convenience function for using [`File::open`] and [`read_to_end`]
/// with fewer imports and without an intermediate variable. It pre-allocates a
/// buffer based on the file size when available, so it is generally faster than
/// reading into a vector created with `Vec::new()`.
///
/// [`File::open`]: struct.File.html#method.open
/// [`read_to_end`]: ../io/trait.Read.html#method.read_to_end
///
/// # Errors
///
/// This function will return an error if `path` does not already exist.
/// Other errors may also be returned according to [`OpenOptions::open`].
///
/// [`OpenOptions::open`]: struct.OpenOptions.html#method.open
///
/// It will also return an error if it encounters while reading an error
/// of a kind other than [`ErrorKind::Interrupted`].
///
/// [`ErrorKind::Interrupted`]: ../../std/io/enum.ErrorKind.html#variant.Interrupted
///
/// # Examples
///
/// ```no_run
/// use std::fs;
/// use std::net::SocketAddr;
///
/// fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
///     let foo: SocketAddr = String::from_utf8_lossy(&fs::read("address.txt")?).parse()?;
///     Ok(())
/// }
/// ```
#[stable(feature = "fs_read_write_bytes", since = "1.26.0")]
pub fn read<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    fn inner(path: &Path) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::with_capacity(initial_buffer_size(&file));
        file.read_to_end(&mut bytes)?;
        Ok(bytes)
    }
    inner(path.as_ref())
}

/// Read the entire contents of a file into a string.
///
/// This is a convenience function for using [`File::open`] and [`read_to_string`]
/// with fewer imports and without an intermediate variable. It pre-allocates a
/// buffer based on the file size when available, so it is generally faster than
/// reading into a string created with `String::new()`.
///
/// [`File::open`]: struct.File.html#method.open
/// [`read_to_string`]: ../io/trait.Read.html#method.read_to_string
///
/// # Errors
///
/// This function will return an error if `path` does not already exist.
/// Other errors may also be returned according to [`OpenOptions::open`].
///
/// [`OpenOptions::open`]: struct.OpenOptions.html#method.open
///
/// It will also return an error if it encounters while reading an error
/// of a kind other than [`ErrorKind::Interrupted`],
/// or if the contents of the file are not valid UTF-8.
///
/// [`ErrorKind::Interrupted`]: ../../std/io/enum.ErrorKind.html#variant.Interrupted
///
/// # Examples
///
/// ```no_run
/// use std::fs;
/// use std::net::SocketAddr;
///
/// fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
///     let foo: SocketAddr = fs::read_to_string("address.txt")?.parse()?;
///     Ok(())
/// }
/// ```
#[stable(feature = "fs_read_write", since = "1.26.0")]
pub fn read_to_string<P: AsRef<Path>>(path: P) -> io::Result<String> {
    fn inner(path: &Path) -> io::Result<String> {
        let mut file = File::open(path)?;
        let mut string = String::with_capacity(initial_buffer_size(&file));
        file.read_to_string(&mut string)?;
        Ok(string)
    }
    inner(path.as_ref())
}

/// Write a slice as the entire contents of a file.
///
/// This function will create a file if it does not exist,
/// and will entirely replace its contents if it does.
///
/// This is a convenience function for using [`File::create`] and [`write_all`]
/// with fewer imports.
///
/// [`File::create`]: struct.File.html#method.create
/// [`write_all`]: ../io/trait.Write.html#method.write_all
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
    /// # Errors
    ///
    /// This function will return an error if `path` does not already exist.
    /// Other errors may also be returned according to [`OpenOptions::open`].
    ///
    /// [`OpenOptions::open`]: struct.OpenOptions.html#method.open
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::open("foo.txt")?;
    ///     Ok(())
    /// }
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
    /// See the [`OpenOptions::open`] function for more details.
    ///
    /// [`OpenOptions::open`]: struct.OpenOptions.html#method.open
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut f = File::create("foo.txt")?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<File> {
        OpenOptions::new().write(true).create(true).truncate(true).open(path.as_ref())
    }

    /// Attempts to sync all OS-internal metadata to disk.
    ///
    /// This function will attempt to ensure that all in-memory data reaches the
    /// filesystem before returning.
    ///
    /// This can be used to handle errors that would otherwise only be caught
    /// when the `File` is closed.  Dropping a file will ignore errors in
    /// synchronizing this in-memory data.
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
    pub fn sync_all(&self) -> io::Result<()> {
        self.inner.fsync()
    }

    /// This function is similar to [`sync_all`], except that it may not
    /// synchronize file metadata to the filesystem.
    ///
    /// This is intended for use cases that must synchronize content, but don't
    /// need the metadata on disk. The goal of this method is to reduce disk
    /// operations.
    ///
    /// Note that some platforms may simply implement this in terms of
    /// [`sync_all`].
    ///
    /// [`sync_all`]: struct.File.html#method.sync_all
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
    /// The file's cursor isn't changed. In particular, if the cursor was at the
    /// end and the file is shrunk using this operation, the cursor will now be
    /// past the end.
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
        Ok(File {
            inner: self.inner.duplicate()?
        })
    }

    /// Changes the permissions on the underlying file.
    ///
    /// # Platform-specific behavior
    ///
    /// This function currently corresponds to the `fchmod` function on Unix and
    /// the `SetFileInformationByHandle` function on Windows. Note that, this
    /// [may change in the future][changes].
    ///
    /// [changes]: ../io/index.html#platform-specific-behavior
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
    #[stable(feature = "set_permissions_atomic", since = "1.16.0")]
    pub fn set_permissions(&self, perm: Permissions) -> io::Result<()> {
        self.inner.set_permissions(perm.0)
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

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for File {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for File {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.write_vectored(bufs)
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
impl Read for &File {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for &File {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.write_vectored(bufs)
    }

    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Seek for &File {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
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
    pub fn write(&mut self, write: bool) -> &mut OpenOptions {
        self.0.write(write); self
    }

    /// Sets the option for the append mode.
    ///
    /// This option, when true, means that writes will append to a file instead
    /// of overwriting previous contents.
    /// Note that setting `.write(true).append(true)` has the same effect as
    /// setting only `.append(true)`.
    ///
    /// For most filesystems, the operating system guarantees that all writes are
    /// atomic: no writes get mangled because another process writes at the same
    /// time.
    ///
    /// One maybe obvious note when using append-mode: make sure that all data
    /// that belongs together is written to the file in one operation. This
    /// can be done by concatenating strings before passing them to [`write()`],
    /// or using a buffered writer (with a buffer of adequate size),
    /// and calling [`flush()`] when the message is complete.
    ///
    /// If a file is opened with both read and append access, beware that after
    /// opening, and after every write, the position for reading may be set at the
    /// end of the file. So, before writing, save the current position (using
    /// [`seek`]`(`[`SeekFrom`]`::`[`Current`]`(0))`), and restore it before the next read.
    ///
    /// ## Note
    ///
    /// This function doesn't create the file if it doesn't exist. Use the [`create`]
    /// method to do so.
    ///
    /// [`write()`]: ../../std/fs/struct.File.html#method.write
    /// [`flush()`]: ../../std/fs/struct.File.html#method.flush
    /// [`seek`]: ../../std/fs/struct.File.html#method.seek
    /// [`SeekFrom`]: ../../std/io/enum.SeekFrom.html
    /// [`Current`]: ../../std/io/enum.SeekFrom.html#variant.Current
    /// [`create`]: #method.create
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().append(true).open("foo.txt");
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
    pub fn truncate(&mut self, truncate: bool) -> &mut OpenOptions {
        self.0.truncate(truncate); self
    }

    /// Sets the option for creating a new file.
    ///
    /// This option indicates whether a new file will be created if the file
    /// does not yet already exist.
    ///
    /// In order for the file to be created, [`write`] or [`append`] access must
    /// be used.
    ///
    /// [`write`]: #method.write
    /// [`append`]: #method.append
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().write(true).create(true).open("foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn create(&mut self, create: bool) -> &mut OpenOptions {
        self.0.create(create); self
    }

    /// Sets the option to always create a new file.
    ///
    /// This option indicates whether a new file will be created.
    /// No file is allowed to exist at the target location, also no (dangling)
    /// symlink.
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
    /// [`.create()`]: #method.create
    /// [`.truncate()`]: #method.truncate
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
    pub fn create_new(&mut self, create_new: bool) -> &mut OpenOptions {
        self.0.create_new(create_new); self
    }

    /// Opens a file at `path` with the options specified by `self`.
    ///
    /// # Errors
    ///
    /// This function will return an error under a number of different
    /// circumstances. Some of these error conditions are listed here, together
    /// with their [`ErrorKind`]. The mapping to [`ErrorKind`]s is not part of
    /// the compatibility contract of the function, especially the `Other` kind
    /// might change to more specific kinds in the future.
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
    /// * [`Other`]: One of the directory components of the specified file path
    ///   was not, in fact, a directory.
    /// * [`Other`]: Filesystem-level errors: full disk, write permission
    ///   requested on a read-only file system, exceeded disk quota, too many
    ///   open files, too long filename, too many symbolic links in the
    ///   specified path (Unix-like systems only), etc.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new().open("foo.txt");
    /// ```
    ///
    /// [`ErrorKind`]: ../io/enum.ErrorKind.html
    /// [`AlreadyExists`]: ../io/enum.ErrorKind.html#variant.AlreadyExists
    /// [`InvalidInput`]: ../io/enum.ErrorKind.html#variant.InvalidInput
    /// [`NotFound`]: ../io/enum.ErrorKind.html#variant.NotFound
    /// [`Other`]: ../io/enum.ErrorKind.html#variant.Other
    /// [`PermissionDenied`]: ../io/enum.ErrorKind.html#variant.PermissionDenied
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn open<P: AsRef<Path>>(&self, path: P) -> io::Result<File> {
        self._open(path.as_ref())
    }

    fn _open(&self, path: &Path) -> io::Result<File> {
        fs_imp::File::open(path, &self.0).map(|inner| File { inner })
    }
}

impl AsInner<fs_imp::OpenOptions> for OpenOptions {
    fn as_inner(&self) -> &fs_imp::OpenOptions { &self.0 }
}

impl AsInnerMut<fs_imp::OpenOptions> for OpenOptions {
    fn as_inner_mut(&mut self) -> &mut fs_imp::OpenOptions { &mut self.0 }
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
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn file_type(&self) -> FileType {
        FileType(self.0.file_type())
    }

    /// Returns `true` if this metadata is for a directory. The
    /// result is mutually exclusive to the result of
    /// [`is_file`], and will be false for symlink metadata
    /// obtained from [`symlink_metadata`].
    ///
    /// [`is_file`]: struct.Metadata.html#method.is_file
    /// [`symlink_metadata`]: fn.symlink_metadata.html
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_dir(&self) -> bool { self.file_type().is_dir() }

    /// Returns `true` if this metadata is for a regular file. The
    /// result is mutually exclusive to the result of
    /// [`is_dir`], and will be false for symlink metadata
    /// obtained from [`symlink_metadata`].
    ///
    /// [`is_dir`]: struct.Metadata.html#method.is_dir
    /// [`symlink_metadata`]: fn.symlink_metadata.html
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_file(&self) -> bool { self.file_type().is_file() }

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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> u64 { self.0.size() }

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
    /// This field may not be available on all platforms, and will return an
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
    ///         println!("{:?}", time);
    ///     } else {
    ///         println!("Not supported on this platform");
    ///     }
    ///     Ok(())
    /// }
    /// ```
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
    /// This field may not be available on all platforms, and will return an
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
    ///         println!("{:?}", time);
    ///     } else {
    ///         println!("Not supported on this platform");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "fs_time", since = "1.10.0")]
    pub fn accessed(&self) -> io::Result<SystemTime> {
        self.0.accessed().map(FromInner::from_inner)
    }

    /// Returns the creation time listed in this metadata.
    ///
    /// The returned value corresponds to the `birthtime` field of `stat` on
    /// Unix platforms and the `ftCreationTime` field on Windows platforms.
    ///
    /// # Errors
    ///
    /// This field may not be available on all platforms, and will return an
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
    ///     if let Ok(time) = metadata.created() {
    ///         println!("{:?}", time);
    ///     } else {
    ///         println!("Not supported on this platform");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "fs_time", since = "1.10.0")]
    pub fn created(&self) -> io::Result<SystemTime> {
        self.0.created().map(FromInner::from_inner)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Metadata")
            .field("file_type", &self.file_type())
            .field("is_dir", &self.is_dir())
            .field("is_file", &self.is_file())
            .field("permissions", &self.permissions())
            .field("modified", &self.modified())
            .field("accessed", &self.accessed())
            .field("created", &self.created())
            .finish()
    }
}

impl AsInner<fs_imp::FileAttr> for Metadata {
    fn as_inner(&self) -> &fs_imp::FileAttr { &self.0 }
}

impl FromInner<fs_imp::FileAttr> for Metadata {
    fn from_inner(attr: fs_imp::FileAttr) -> Metadata { Metadata(attr) }
}

impl Permissions {
    /// Returns `true` if these permissions describe a readonly (unwritable) file.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn readonly(&self) -> bool { self.0.readonly() }

    /// Modifies the readonly flag for this set of permissions. If the
    /// `readonly` argument is `true`, using the resulting `Permission` will
    /// update file permissions to forbid writing. Conversely, if it's `false`,
    /// using the resulting `Permission` will update file permissions to allow
    /// writing.
    ///
    /// This operation does **not** modify the filesystem. To modify the
    /// filesystem use the [`fs::set_permissions`] function.
    ///
    /// [`fs::set_permissions`]: fn.set_permissions.html
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
    ///     // filesystem doesn't change
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
    /// [`is_file`]: struct.FileType.html#method.is_file
    /// [`is_symlink`]: struct.FileType.html#method.is_symlink
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
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn is_dir(&self) -> bool { self.0.is_dir() }

    /// Tests whether this file type represents a regular file.
    /// The result is  mutually exclusive to the results of
    /// [`is_dir`] and [`is_symlink`]; only zero or one of these
    /// tests may pass.
    ///
    /// [`is_dir`]: struct.FileType.html#method.is_dir
    /// [`is_symlink`]: struct.FileType.html#method.is_symlink
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
    #[stable(feature = "file_type", since = "1.1.0")]
    pub fn is_file(&self) -> bool { self.0.is_file() }

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
    /// [`Metadata`]: struct.Metadata.html
    /// [`fs::metadata`]: fn.metadata.html
    /// [`fs::symlink_metadata`]: fn.symlink_metadata.html
    /// [`is_dir`]: struct.FileType.html#method.is_dir
    /// [`is_file`]: struct.FileType.html#method.is_file
    /// [`is_symlink`]: struct.FileType.html#method.is_symlink
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn path(&self) -> PathBuf { self.0.path() }

    /// Returns the metadata for the file that this entry points at.
    ///
    /// This function will not traverse symlinks if this entry points at a
    /// symlink.
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

    /// Returns the bare file name of this directory entry without any other
    /// leading path component.
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
    #[stable(feature = "dir_entry_ext", since = "1.1.0")]
    pub fn file_name(&self) -> OsString {
        self.0.file_name()
    }
}

#[stable(feature = "dir_entry_debug", since = "1.13.0")]
impl fmt::Debug for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DirEntry")
            .field(&self.path())
            .finish()
    }
}

impl AsInner<fs_imp::DirEntry> for DirEntry {
    fn as_inner(&self) -> &fs_imp::DirEntry { &self.0 }
}

/// Removes a file from the filesystem.
///
/// Note that there is no
/// guarantee that the file is immediately deleted (e.g., depending on
/// platform, other open file descriptors may prevent immediate removal).
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `unlink` function on Unix
/// and the `DeleteFile` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * `path` points to a directory.
/// * The user lacks permissions to remove the file.
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
/// # Platform-specific behavior
///
/// This function currently corresponds to the `stat` function on Unix
/// and the `GetFileAttributesEx` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn metadata<P: AsRef<Path>>(path: P) -> io::Result<Metadata> {
    fs_imp::stat(path.as_ref()).map(Metadata)
}

/// Query the metadata about a file without following symlinks.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `lstat` function on Unix
/// and the `GetFileAttributesEx` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
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
#[stable(feature = "symlink_metadata", since = "1.1.0")]
pub fn symlink_metadata<P: AsRef<Path>>(path: P) -> io::Result<Metadata> {
    fs_imp::lstat(path.as_ref()).map(Metadata)
}

/// Rename a file or directory to a new name, replacing the original file if
/// `to` already exists.
///
/// This will not work if the new name is on a different mount point.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `rename` function on Unix
/// and the `MoveFileEx` function with the `MOVEFILE_REPLACE_EXISTING` flag on Windows.
///
/// Because of this, the behavior when both `from` and `to` exist differs. On
/// Unix, if `from` is a directory, `to` must also be an (empty) directory. If
/// `from` is not a directory, `to` must also be not a directory. In contrast,
/// on Windows, `from` can be anything, but `to` must *not* be a directory.
///
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
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
/// If you’re wanting to copy the contents of one file to another and you’re
/// working with [`File`]s, see the [`io::copy`] function.
///
/// [`io::copy`]: ../io/fn.copy.html
/// [`File`]: ./struct.File.html
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `open` function in Unix
/// with `O_RDONLY` for `from` and `O_WRONLY`, `O_CREAT`, and `O_TRUNC` for `to`.
/// `O_CLOEXEC` is set for returned file descriptors.
/// On Windows, this function currently corresponds to `CopyFileEx`. Alternate
/// NTFS streams are copied but only the size of the main stream is returned by
/// this function. On MacOS, this function corresponds to `fclonefileat` and
/// `fcopyfile`.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The `from` path is not a file.
/// * The `from` file does not exist.
/// * The current process does not have the permission rights to access
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
    fs_imp::copy(from.as_ref(), to.as_ref())
}

/// Creates a new hard link on the filesystem.
///
/// The `dst` path will be a link pointing to the `src` path. Note that systems
/// often require these two paths to both be located on the same filesystem.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `link` function on Unix
/// and the `CreateHardLink` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The `src` path is not a file or doesn't exist.
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn hard_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> io::Result<()> {
    fs_imp::link(src.as_ref(), dst.as_ref())
}

/// Creates a new symbolic link on the filesystem.
///
/// The `dst` path will be a symbolic link pointing to the `src` path.
/// On Windows, this will be a file symlink, not a directory symlink;
/// for this reason, the platform-specific [`std::os::unix::fs::symlink`]
/// and [`std::os::windows::fs::symlink_file`] or [`symlink_dir`] should be
/// used instead to make the intent explicit.
///
/// [`std::os::unix::fs::symlink`]: ../os/unix/fs/fn.symlink.html
/// [`std::os::windows::fs::symlink_file`]: ../os/windows/fs/fn.symlink_file.html
/// [`symlink_dir`]: ../os/windows/fs/fn.symlink_dir.html
///
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
#[rustc_deprecated(since = "1.1.0",
             reason = "replaced with std::os::unix::fs::symlink and \
                       std::os::windows::fs::{symlink_file, symlink_dir}")]
pub fn soft_link<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> io::Result<()> {
    fs_imp::symlink(src.as_ref(), dst.as_ref())
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
/// [changes]: ../io/index.html#platform-specific-behavior
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
/// Note that, this [may change in the future][changes].
///
/// On Windows, this converts the path to use [extended length path][path]
/// syntax, which allows your program to use longer path names, but means you
/// can only join backslash-delimited paths to it, and it may be incompatible
/// with other applications (if passed to the application on the command-line,
/// or written to a file another application may read).
///
/// [changes]: ../io/index.html#platform-specific-behavior
/// [path]: https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx#maxpath
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
#[stable(feature = "fs_canonicalize", since = "1.5.0")]
pub fn canonicalize<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    fs_imp::canonicalize(path.as_ref())
}

/// Creates a new, empty directory at the provided path
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `mkdir` function on Unix
/// and the `CreateDirectory` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
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
/// [`create_dir_all`]: fn.create_dir_all.html
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    DirBuilder::new().create(path.as_ref())
}

/// Recursively create a directory and all of its parent components if they
/// are missing.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `mkdir` function on Unix
/// and the `CreateDirectory` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * If any directory in the path specified by `path`
/// does not already exist and it could not be created otherwise. The specific
/// error conditions for when a directory is being created (after it is
/// determined to not exist) are outlined by [`fs::create_dir`].
///
/// Notable exception is made for situations where any of the directories
/// specified in the `path` could not be created as it was being created concurrently.
/// Such cases are considered to be successful. That is, calling `create_dir_all`
/// concurrently from multiple threads or processes is guaranteed not to fail
/// due to a race condition with itself.
///
/// [`fs::create_dir`]: fn.create_dir.html
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

/// Removes an existing, empty directory.
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `rmdir` function on Unix
/// and the `RemoveDirectory` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// * The user lacks permissions to remove the directory at the provided `path`.
/// * The directory isn't empty.
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
/// This function currently corresponds to `opendir`, `lstat`, `rm` and `rmdir` functions on Unix
/// and the `FindFirstFile`, `GetFileAttributesEx`, `DeleteFile`, and `RemoveDirectory` functions
/// on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
///
/// # Errors
///
/// See [`fs::remove_file`] and [`fs::remove_dir`].
///
/// [`fs::remove_file`]:  fn.remove_file.html
/// [`fs::remove_dir`]: fn.remove_dir.html
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
/// The iterator will yield instances of [`io::Result`]`<`[`DirEntry`]`>`.
/// New errors may be encountered after an iterator is initially constructed.
///
/// [`io::Result`]: ../io/type.Result.html
/// [`DirEntry`]: struct.DirEntry.html
///
/// # Platform-specific behavior
///
/// This function currently corresponds to the `opendir` function on Unix
/// and the `FindFirstFile` function on Windows.
/// Note that, this [may change in the future][changes].
///
/// [changes]: ../io/index.html#platform-specific-behavior
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
/// [changes]: ../io/index.html#platform-specific-behavior
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
#[stable(feature = "set_permissions", since = "1.1.0")]
pub fn set_permissions<P: AsRef<Path>>(path: P, perm: Permissions)
                                       -> io::Result<()> {
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
    pub fn new() -> DirBuilder {
        DirBuilder {
            inner: fs_imp::DirBuilder::new(),
            recursive: false,
        }
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
        if self.recursive {
            self.create_dir_all(path)
        } else {
            self.inner.mkdir(path)
        }
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        if path == Path::new("") {
            return Ok(())
        }

        match self.inner.mkdir(path) {
            Ok(()) => return Ok(()),
            Err(ref e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(_) if path.is_dir() => return Ok(()),
            Err(e) => return Err(e),
        }
        match path.parent() {
            Some(p) => self.create_dir_all(p)?,
            None => return Err(io::Error::new(io::ErrorKind::Other, "failed to create whole tree")),
        }
        match self.inner.mkdir(path) {
            Ok(()) => Ok(()),
            Err(_) if path.is_dir() => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl AsInnerMut<fs_imp::DirBuilder> for DirBuilder {
    fn as_inner_mut(&mut self) -> &mut fs_imp::DirBuilder {
        &mut self.inner
    }
}

#[cfg(all(test, not(any(target_os = "cloudabi", target_os = "emscripten", target_env = "sgx"))))]
mod tests {
    use crate::io::prelude::*;

    use crate::fs::{self, File, OpenOptions};
    use crate::io::{ErrorKind, SeekFrom};
    use crate::path::Path;
    use crate::str;
    use crate::sys_common::io::test::{TempDir, tmpdir};
    use crate::thread;

    use rand::{rngs::StdRng, FromEntropy, RngCore};

    #[cfg(windows)]
    use crate::os::windows::fs::{symlink_dir, symlink_file};
    #[cfg(windows)]
    use crate::sys::fs::symlink_junction;
    #[cfg(unix)]
    use crate::os::unix::fs::symlink as symlink_dir;
    #[cfg(unix)]
    use crate::os::unix::fs::symlink as symlink_file;
    #[cfg(unix)]
    use crate::os::unix::fs::symlink as symlink_junction;

    macro_rules! check { ($e:expr) => (
        match $e {
            Ok(t) => t,
            Err(e) => panic!("{} failed with: {}", stringify!($e), e),
        }
    ) }

    #[cfg(windows)]
    macro_rules! error { ($e:expr, $s:expr) => (
        match $e {
            Ok(_) => panic!("Unexpected success. Should've been: {:?}", $s),
            Err(ref err) => assert!(err.raw_os_error() == Some($s),
                                    format!("`{}` did not have a code of `{}`", err, $s))
        }
    ) }

    #[cfg(unix)]
    macro_rules! error { ($e:expr, $s:expr) => ( error_contains!($e, $s) ) }

    macro_rules! error_contains { ($e:expr, $s:expr) => (
        match $e {
            Ok(_) => panic!("Unexpected success. Should've been: {:?}", $s),
            Err(ref err) => assert!(err.to_string().contains($s),
                                    format!("`{}` did not contain `{}`", err, $s))
        }
    ) }

    // Several test fail on windows if the user does not have permission to
    // create symlinks (the `SeCreateSymbolicLinkPrivilege`). Instead of
    // disabling these test on Windows, use this function to test whether we
    // have permission, and return otherwise. This way, we still don't run these
    // tests most of the time, but at least we do if the user has the right
    // permissions.
    pub fn got_symlink_permission(tmpdir: &TempDir) -> bool {
        if cfg!(unix) { return true }
        let link = tmpdir.join("some_hopefully_unique_link_name");

        match symlink_file(r"nonexisting_target", link) {
            Ok(_) => true,
            // ERROR_PRIVILEGE_NOT_HELD = 1314
            Err(ref err) if err.raw_os_error() == Some(1314) => false,
            Err(_) => true,
        }
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

        #[cfg(unix)]
        error!(result, "No such file or directory");
        #[cfg(windows)]
        error!(result, 2); // ERROR_FILE_NOT_FOUND
    }

    #[test]
    fn file_test_iounlinking_invalid_path_should_raise_condition() {
        let tmpdir = tmpdir();
        let filename = &tmpdir.join("file_another_file_that_does_not_exist.txt");

        let result = fs::remove_file(filename);

        #[cfg(unix)]
        error!(result, "No such file or directory");
        #[cfg(windows)]
        error!(result, 2); // ERROR_FILE_NOT_FOUND
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
        let tell_pos_pre_read;
        let tell_pos_post_read;
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
    fn file_test_io_eof() {
        let tmpdir = tmpdir();
        let filename = tmpdir.join("file_rt_io_file_test_eof.txt");
        let mut buf = [0; 256];
        {
            let oo = OpenOptions::new().create_new(true).write(true).read(true).clone();
            let mut rw = check!(oo.open(&filename));
            assert_eq!(check!(rw.read(&mut buf)), 0);
            assert_eq!(check!(rw.read(&mut buf)), 0);
        }
        check!(fs::remove_file(&filename));
    }

    #[test]
    #[cfg(unix)]
    fn file_test_io_read_write_at() {
        use crate::os::unix::fs::FileExt;

        let tmpdir = tmpdir();
        let filename = tmpdir.join("file_rt_io_file_test_read_write_at.txt");
        let mut buf = [0; 256];
        let write1 = "asdf";
        let write2 = "qwer-";
        let write3 = "-zxcv";
        let content = "qwer-asdf-zxcv";
        {
            let oo = OpenOptions::new().create_new(true).write(true).read(true).clone();
            let mut rw = check!(oo.open(&filename));
            assert_eq!(check!(rw.write_at(write1.as_bytes(), 5)), write1.len());
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 0);
            assert_eq!(check!(rw.read_at(&mut buf, 5)), write1.len());
            assert_eq!(str::from_utf8(&buf[..write1.len()]), Ok(write1));
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 0);
            assert_eq!(check!(rw.read_at(&mut buf[..write2.len()], 0)), write2.len());
            assert_eq!(str::from_utf8(&buf[..write2.len()]), Ok("\0\0\0\0\0"));
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 0);
            assert_eq!(check!(rw.write(write2.as_bytes())), write2.len());
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 5);
            assert_eq!(check!(rw.read(&mut buf)), write1.len());
            assert_eq!(str::from_utf8(&buf[..write1.len()]), Ok(write1));
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 9);
            assert_eq!(check!(rw.read_at(&mut buf[..write2.len()], 0)), write2.len());
            assert_eq!(str::from_utf8(&buf[..write2.len()]), Ok(write2));
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 9);
            assert_eq!(check!(rw.write_at(write3.as_bytes(), 9)), write3.len());
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 9);
        }
        {
            let mut read = check!(File::open(&filename));
            assert_eq!(check!(read.read_at(&mut buf, 0)), content.len());
            assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 0);
            assert_eq!(check!(read.seek(SeekFrom::End(-5))), 9);
            assert_eq!(check!(read.read_at(&mut buf, 0)), content.len());
            assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 9);
            assert_eq!(check!(read.read(&mut buf)), write3.len());
            assert_eq!(str::from_utf8(&buf[..write3.len()]), Ok(write3));
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 14);
            assert_eq!(check!(read.read_at(&mut buf, 0)), content.len());
            assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 14);
            assert_eq!(check!(read.read_at(&mut buf, 14)), 0);
            assert_eq!(check!(read.read_at(&mut buf, 15)), 0);
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 14);
        }
        check!(fs::remove_file(&filename));
    }

    #[test]
    #[cfg(unix)]
    fn set_get_unix_permissions() {
        use crate::os::unix::fs::PermissionsExt;

        let tmpdir = tmpdir();
        let filename = &tmpdir.join("set_get_unix_permissions");
        check!(fs::create_dir(filename));
        let mask = 0o7777;

        check!(fs::set_permissions(filename,
                                   fs::Permissions::from_mode(0)));
        let metadata0 = check!(fs::metadata(filename));
        assert_eq!(mask & metadata0.permissions().mode(), 0);

        check!(fs::set_permissions(filename,
                                   fs::Permissions::from_mode(0o1777)));
        let metadata1 = check!(fs::metadata(filename));
        assert_eq!(mask & metadata1.permissions().mode(), 0o1777);
    }

    #[test]
    #[cfg(windows)]
    fn file_test_io_seek_read_write() {
        use crate::os::windows::fs::FileExt;

        let tmpdir = tmpdir();
        let filename = tmpdir.join("file_rt_io_file_test_seek_read_write.txt");
        let mut buf = [0; 256];
        let write1 = "asdf";
        let write2 = "qwer-";
        let write3 = "-zxcv";
        let content = "qwer-asdf-zxcv";
        {
            let oo = OpenOptions::new().create_new(true).write(true).read(true).clone();
            let mut rw = check!(oo.open(&filename));
            assert_eq!(check!(rw.seek_write(write1.as_bytes(), 5)), write1.len());
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 9);
            assert_eq!(check!(rw.seek_read(&mut buf, 5)), write1.len());
            assert_eq!(str::from_utf8(&buf[..write1.len()]), Ok(write1));
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 9);
            assert_eq!(check!(rw.seek(SeekFrom::Start(0))), 0);
            assert_eq!(check!(rw.write(write2.as_bytes())), write2.len());
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 5);
            assert_eq!(check!(rw.read(&mut buf)), write1.len());
            assert_eq!(str::from_utf8(&buf[..write1.len()]), Ok(write1));
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 9);
            assert_eq!(check!(rw.seek_read(&mut buf[..write2.len()], 0)), write2.len());
            assert_eq!(str::from_utf8(&buf[..write2.len()]), Ok(write2));
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 5);
            assert_eq!(check!(rw.seek_write(write3.as_bytes(), 9)), write3.len());
            assert_eq!(check!(rw.seek(SeekFrom::Current(0))), 14);
        }
        {
            let mut read = check!(File::open(&filename));
            assert_eq!(check!(read.seek_read(&mut buf, 0)), content.len());
            assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 14);
            assert_eq!(check!(read.seek(SeekFrom::End(-5))), 9);
            assert_eq!(check!(read.seek_read(&mut buf, 0)), content.len());
            assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 14);
            assert_eq!(check!(read.seek(SeekFrom::End(-5))), 9);
            assert_eq!(check!(read.read(&mut buf)), write3.len());
            assert_eq!(str::from_utf8(&buf[..write3.len()]), Ok(write3));
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 14);
            assert_eq!(check!(read.seek_read(&mut buf, 0)), content.len());
            assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
            assert_eq!(check!(read.seek(SeekFrom::Current(0))), 14);
            assert_eq!(check!(read.seek_read(&mut buf, 14)), 0);
            assert_eq!(check!(read.seek_read(&mut buf, 15)), 0);
        }
        check!(fs::remove_file(&filename));
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
        assert!(!dir.is_file());
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
    fn file_create_new_already_exists_error() {
        let tmpdir = tmpdir();
        let file = &tmpdir.join("file_create_new_error_exists");
        check!(fs::File::create(file));
        let e = fs::OpenOptions::new().write(true).create_new(true).open(file).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::AlreadyExists);
    }

    #[test]
    fn mkdir_path_already_exists_error() {
        let tmpdir = tmpdir();
        let dir = &tmpdir.join("mkdir_error_twice");
        check!(fs::create_dir(dir));
        let e = fs::create_dir(dir).unwrap_err();
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
    }

    #[test]
    fn concurrent_recursive_mkdir() {
        for _ in 0..100 {
            let dir = tmpdir();
            let mut dir = dir.join("a");
            for _ in 0..40 {
                dir = dir.join("a");
            }
            let mut join = vec!();
            for _ in 0..8 {
                let dir = dir.clone();
                join.push(thread::spawn(move || {
                    check!(fs::create_dir_all(&dir));
                }))
            }

            // No `Display` on result of `join()`
            join.drain(..).map(|join| join.join().unwrap()).count();
        }
    }

    #[test]
    fn recursive_mkdir_slash() {
        check!(fs::create_dir_all(Path::new("/")));
    }

    #[test]
    fn recursive_mkdir_dot() {
        check!(fs::create_dir_all(Path::new(".")));
    }

    #[test]
    fn recursive_mkdir_empty() {
        check!(fs::create_dir_all(Path::new("")));
    }

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
        check!(symlink_junction(&d2, &dt.join("d2")));
        let _ = symlink_file(&canary, &d1.join("canary"));
        check!(fs::remove_dir_all(&d1));

        assert!(!d1.is_dir());
        assert!(canary.exists());
    }

    #[test]
    fn recursive_rmdir_of_symlink() {
        // test we do not recursively delete a symlink but only dirs.
        let tmpdir = tmpdir();
        let link = tmpdir.join("d1");
        let dir = tmpdir.join("d2");
        let canary = dir.join("do_not_delete");
        check!(fs::create_dir_all(&dir));
        check!(check!(File::create(&canary)).write(b"foo"));
        check!(symlink_junction(&dir, &link));
        check!(fs::remove_dir_all(&link));

        assert!(!link.is_dir());
        assert!(canary.exists());
    }

    #[test]
    // only Windows makes a distinction between file and directory symlinks.
    #[cfg(windows)]
    fn recursive_rmdir_of_file_symlink() {
        let tmpdir = tmpdir();
        if !got_symlink_permission(&tmpdir) { return };

        let f1 = tmpdir.join("f1");
        let f2 = tmpdir.join("f2");
        check!(check!(File::create(&f1)).write(b"foo"));
        check!(symlink_file(&f1, &f2));
        match fs::remove_dir_all(&f2) {
            Ok(..) => panic!("wanted a failure"),
            Err(..) => {}
        }
    }

    #[test]
    fn unicode_path_is_dir() {
        assert!(Path::new(".").is_dir());
        assert!(!Path::new("test/stdtest/fs.rs").is_dir());

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
        assert!(Path::new(".").exists());
        assert!(!Path::new("test/nonexistent-bogus-path").exists());

        let tmpdir = tmpdir();
        let unicode = tmpdir.path();
        let unicode = unicode.join("test-각丁ー再见");
        check!(fs::create_dir(&unicode));
        assert!(unicode.exists());
        assert!(!Path::new("test/unicode-bogus-path-각丁ー再见").exists());
    }

    #[test]
    fn copy_file_does_not_exist() {
        let from = Path::new("test/nonexistent-bogus-path");
        let to = Path::new("test/other-bogus-path");

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
        let from = Path::new("test/nonexistent-bogus-path");
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

    #[test]
    #[cfg(windows)]
    fn copy_file_preserves_streams() {
        let tmp = tmpdir();
        check!(check!(File::create(tmp.join("in.txt:bunny"))).write("carrot".as_bytes()));
        assert_eq!(check!(fs::copy(tmp.join("in.txt"), tmp.join("out.txt"))), 0);
        assert_eq!(check!(tmp.join("out.txt").metadata()).len(), 0);
        let mut v = Vec::new();
        check!(check!(File::open(tmp.join("out.txt:bunny"))).read_to_end(&mut v));
        assert_eq!(v, b"carrot".to_vec());
    }

    #[test]
    fn copy_file_returns_metadata_len() {
        let tmp = tmpdir();
        let in_path = tmp.join("in.txt");
        let out_path = tmp.join("out.txt");
        check!(check!(File::create(&in_path)).write(b"lettuce"));
        #[cfg(windows)]
        check!(check!(File::create(tmp.join("in.txt:bunny"))).write(b"carrot"));
        let copied_len = check!(fs::copy(&in_path, &out_path));
        assert_eq!(check!(out_path.metadata()).len(), copied_len);
    }

    #[test]
    fn copy_file_follows_dst_symlink() {
        let tmp = tmpdir();
        if !got_symlink_permission(&tmp) { return };

        let in_path = tmp.join("in.txt");
        let out_path = tmp.join("out.txt");
        let out_path_symlink = tmp.join("out_symlink.txt");

        check!(fs::write(&in_path, "foo"));
        check!(fs::write(&out_path, "bar"));
        check!(symlink_file(&out_path, &out_path_symlink));

        check!(fs::copy(&in_path, &out_path_symlink));

        assert!(check!(out_path_symlink.symlink_metadata()).file_type().is_symlink());
        assert_eq!(check!(fs::read(&out_path_symlink)), b"foo".to_vec());
        assert_eq!(check!(fs::read(&out_path)), b"foo".to_vec());
    }

    #[test]
    fn symlinks_work() {
        let tmpdir = tmpdir();
        if !got_symlink_permission(&tmpdir) { return };

        let input = tmpdir.join("in.txt");
        let out = tmpdir.join("out.txt");

        check!(check!(File::create(&input)).write("foobar".as_bytes()));
        check!(symlink_file(&input, &out));
        assert!(check!(out.symlink_metadata()).file_type().is_symlink());
        assert_eq!(check!(fs::metadata(&out)).len(),
                   check!(fs::metadata(&input)).len());
        let mut v = Vec::new();
        check!(check!(File::open(&out)).read_to_end(&mut v));
        assert_eq!(v, b"foobar".to_vec());
    }

    #[test]
    fn symlink_noexist() {
        // Symlinks can point to things that don't exist
        let tmpdir = tmpdir();
        if !got_symlink_permission(&tmpdir) { return };

        // Use a relative path for testing. Symlinks get normalized by Windows,
        // so we may not get the same path back for absolute paths
        check!(symlink_file(&"foo", &tmpdir.join("bar")));
        assert_eq!(check!(fs::read_link(&tmpdir.join("bar"))).to_str().unwrap(),
                   "foo");
    }

    #[test]
    fn read_link() {
        if cfg!(windows) {
            // directory symlink
            assert_eq!(check!(fs::read_link(r"C:\Users\All Users")).to_str().unwrap(),
                       r"C:\ProgramData");
            // junction
            assert_eq!(check!(fs::read_link(r"C:\Users\Default User")).to_str().unwrap(),
                       r"C:\Users\Default");
            // junction with special permissions
            assert_eq!(check!(fs::read_link(r"C:\Documents and Settings\")).to_str().unwrap(),
                       r"C:\Users");
        }
        let tmpdir = tmpdir();
        let link = tmpdir.join("link");
        if !got_symlink_permission(&tmpdir) { return };
        check!(symlink_file(&"foo", &link));
        assert_eq!(check!(fs::read_link(&link)).to_str().unwrap(), "foo");
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
    fn fchmod_works() {
        let tmpdir = tmpdir();
        let path = tmpdir.join("in.txt");

        let file = check!(File::create(&path));
        let attr = check!(fs::metadata(&path));
        assert!(!attr.permissions().readonly());
        let mut p = attr.permissions();
        p.set_readonly(true);
        check!(file.set_permissions(p.clone()));
        let attr = check!(fs::metadata(&path));
        assert!(attr.permissions().readonly());

        p.set_readonly(false);
        check!(file.set_permissions(p));
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
        use crate::fs::OpenOptions as OO;
        fn c<T: Clone>(t: &T) -> T { t.clone() }

        let tmpdir = tmpdir();

        let mut r = OO::new(); r.read(true);
        let mut w = OO::new(); w.write(true);
        let mut rw = OO::new(); rw.read(true).write(true);
        let mut a = OO::new(); a.append(true);
        let mut ra = OO::new(); ra.read(true).append(true);

        #[cfg(windows)]
        let invalid_options = 87; // ERROR_INVALID_PARAMETER
        #[cfg(unix)]
        let invalid_options = "Invalid argument";

        // Test various combinations of creation modes and access modes.
        //
        // Allowed:
        // creation mode           | read  | write | read-write | append | read-append |
        // :-----------------------|:-----:|:-----:|:----------:|:------:|:-----------:|
        // not set (open existing) |   X   |   X   |     X      |   X    |      X      |
        // create                  |       |   X   |     X      |   X    |      X      |
        // truncate                |       |   X   |     X      |        |             |
        // create and truncate     |       |   X   |     X      |        |             |
        // create_new              |       |   X   |     X      |   X    |      X      |
        //
        // tested in reverse order, so 'create_new' creates the file, and 'open existing' opens it.

        // write-only
        check!(c(&w).create_new(true).open(&tmpdir.join("a")));
        check!(c(&w).create(true).truncate(true).open(&tmpdir.join("a")));
        check!(c(&w).truncate(true).open(&tmpdir.join("a")));
        check!(c(&w).create(true).open(&tmpdir.join("a")));
        check!(c(&w).open(&tmpdir.join("a")));

        // read-only
        error!(c(&r).create_new(true).open(&tmpdir.join("b")), invalid_options);
        error!(c(&r).create(true).truncate(true).open(&tmpdir.join("b")), invalid_options);
        error!(c(&r).truncate(true).open(&tmpdir.join("b")), invalid_options);
        error!(c(&r).create(true).open(&tmpdir.join("b")), invalid_options);
        check!(c(&r).open(&tmpdir.join("a"))); // try opening the file created with write_only

        // read-write
        check!(c(&rw).create_new(true).open(&tmpdir.join("c")));
        check!(c(&rw).create(true).truncate(true).open(&tmpdir.join("c")));
        check!(c(&rw).truncate(true).open(&tmpdir.join("c")));
        check!(c(&rw).create(true).open(&tmpdir.join("c")));
        check!(c(&rw).open(&tmpdir.join("c")));

        // append
        check!(c(&a).create_new(true).open(&tmpdir.join("d")));
        error!(c(&a).create(true).truncate(true).open(&tmpdir.join("d")), invalid_options);
        error!(c(&a).truncate(true).open(&tmpdir.join("d")), invalid_options);
        check!(c(&a).create(true).open(&tmpdir.join("d")));
        check!(c(&a).open(&tmpdir.join("d")));

        // read-append
        check!(c(&ra).create_new(true).open(&tmpdir.join("e")));
        error!(c(&ra).create(true).truncate(true).open(&tmpdir.join("e")), invalid_options);
        error!(c(&ra).truncate(true).open(&tmpdir.join("e")), invalid_options);
        check!(c(&ra).create(true).open(&tmpdir.join("e")));
        check!(c(&ra).open(&tmpdir.join("e")));

        // Test opening a file without setting an access mode
        let mut blank = OO::new();
         error!(blank.create(true).open(&tmpdir.join("f")), invalid_options);

        // Test write works
        check!(check!(File::create(&tmpdir.join("h"))).write("foobar".as_bytes()));

        // Test write fails for read-only
        check!(r.open(&tmpdir.join("h")));
        {
            let mut f = check!(r.open(&tmpdir.join("h")));
            assert!(f.write("wut".as_bytes()).is_err());
        }

        // Test write overwrites
        {
            let mut f = check!(c(&w).open(&tmpdir.join("h")));
            check!(f.write("baz".as_bytes()));
        }
        {
            let mut f = check!(c(&r).open(&tmpdir.join("h")));
            let mut b = vec![0; 6];
            check!(f.read(&mut b));
            assert_eq!(b, "bazbar".as_bytes());
        }

        // Test truncate works
        {
            let mut f = check!(c(&w).truncate(true).open(&tmpdir.join("h")));
            check!(f.write("foo".as_bytes()));
        }
        assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 3);

        // Test append works
        assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 3);
        {
            let mut f = check!(c(&a).open(&tmpdir.join("h")));
            check!(f.write("bar".as_bytes()));
        }
        assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 6);

        // Test .append(true) equals .write(true).append(true)
        {
            let mut f = check!(c(&w).append(true).open(&tmpdir.join("h")));
            check!(f.write("baz".as_bytes()));
        }
        assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 9);
    }

    #[test]
    fn _assert_send_sync() {
        fn _assert_send_sync<T: Send + Sync>() {}
        _assert_send_sync::<OpenOptions>();
    }

    #[test]
    fn binary_file() {
        let mut bytes = [0; 1024];
        StdRng::from_entropy().fill_bytes(&mut bytes);

        let tmpdir = tmpdir();

        check!(check!(File::create(&tmpdir.join("test"))).write(&bytes));
        let mut v = Vec::new();
        check!(check!(File::open(&tmpdir.join("test"))).read_to_end(&mut v));
        assert!(v == &bytes[..]);
    }

    #[test]
    fn write_then_read() {
        let mut bytes = [0; 1024];
        StdRng::from_entropy().fill_bytes(&mut bytes);

        let tmpdir = tmpdir();

        check!(fs::write(&tmpdir.join("test"), &bytes[..]));
        let v = check!(fs::read(&tmpdir.join("test")));
        assert!(v == &bytes[..]);

        check!(fs::write(&tmpdir.join("not-utf8"), &[0xFF]));
        error_contains!(fs::read_to_string(&tmpdir.join("not-utf8")),
                        "stream did not contain valid UTF-8");

        let s = "𐁁𐀓𐀠𐀴𐀍";
        check!(fs::write(&tmpdir.join("utf8"), s.as_bytes()));
        let string = check!(fs::read_to_string(&tmpdir.join("utf8")));
        assert_eq!(string, s);
    }

    #[test]
    fn file_try_clone() {
        let tmpdir = tmpdir();

        let mut f1 = check!(OpenOptions::new()
                                       .read(true)
                                       .write(true)
                                       .create(true)
                                       .open(&tmpdir.join("test")));
        let mut f2 = check!(f1.try_clone());

        check!(f1.write_all(b"hello world"));
        check!(f1.seek(SeekFrom::Start(2)));

        let mut buf = vec![];
        check!(f2.read_to_end(&mut buf));
        assert_eq!(buf, b"llo world");
        drop(f2);

        check!(f1.write_all(b"!"));
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
    fn realpath_works() {
        let tmpdir = tmpdir();
        if !got_symlink_permission(&tmpdir) { return };

        let tmpdir = fs::canonicalize(tmpdir.path()).unwrap();
        let file = tmpdir.join("test");
        let dir = tmpdir.join("test2");
        let link = dir.join("link");
        let linkdir = tmpdir.join("test3");

        File::create(&file).unwrap();
        fs::create_dir(&dir).unwrap();
        symlink_file(&file, &link).unwrap();
        symlink_dir(&dir, &linkdir).unwrap();

        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());

        assert_eq!(fs::canonicalize(&tmpdir).unwrap(), tmpdir);
        assert_eq!(fs::canonicalize(&file).unwrap(), file);
        assert_eq!(fs::canonicalize(&link).unwrap(), file);
        assert_eq!(fs::canonicalize(&linkdir).unwrap(), dir);
        assert_eq!(fs::canonicalize(&linkdir.join("link")).unwrap(), file);
    }

    #[test]
    fn realpath_works_tricky() {
        let tmpdir = tmpdir();
        if !got_symlink_permission(&tmpdir) { return };

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
        if cfg!(not(windows)) {
            symlink_dir("../d/e", &c).unwrap();
            symlink_file("../f", &e).unwrap();
        }
        if cfg!(windows) {
            symlink_dir(r"..\d\e", &c).unwrap();
            symlink_file(r"..\f", &e).unwrap();
        }

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
    fn dir_entry_debug() {
        let tmpdir = tmpdir();
        File::create(&tmpdir.join("b")).unwrap();
        let mut read_dir = tmpdir.path().read_dir().unwrap();
        let dir_entry = read_dir.next().unwrap().unwrap();
        let actual = format!("{:?}", dir_entry);
        let expected = format!("DirEntry({:?})", dir_entry.0.path());
        assert_eq!(actual, expected);
    }

    #[test]
    fn read_dir_not_found() {
        let res = fs::read_dir("/path/that/does/not/exist");
        assert_eq!(res.err().unwrap().kind(), ErrorKind::NotFound);
    }

    #[test]
    fn create_dir_all_with_junctions() {
        let tmpdir = tmpdir();
        let target = tmpdir.join("target");

        let junction = tmpdir.join("junction");
        let b = junction.join("a/b");

        let link = tmpdir.join("link");
        let d = link.join("c/d");

        fs::create_dir(&target).unwrap();

        check!(symlink_junction(&target, &junction));
        check!(fs::create_dir_all(&b));
        // the junction itself is not a directory, but `is_dir()` on a Path
        // follows links
        assert!(junction.is_dir());
        assert!(b.exists());

        if !got_symlink_permission(&tmpdir) { return };
        check!(symlink_dir(&target, &link));
        check!(fs::create_dir_all(&d));
        assert!(link.is_dir());
        assert!(d.exists());
    }

    #[test]
    fn metadata_access_times() {
        let tmpdir = tmpdir();

        let b = tmpdir.join("b");
        File::create(&b).unwrap();

        let a = check!(fs::metadata(&tmpdir.path()));
        let b = check!(fs::metadata(&b));

        assert_eq!(check!(a.accessed()), check!(a.accessed()));
        assert_eq!(check!(a.modified()), check!(a.modified()));
        assert_eq!(check!(b.accessed()), check!(b.modified()));

        if cfg!(target_os = "macos") || cfg!(target_os = "windows") {
            check!(a.created());
            check!(b.created());
        }
    }
}
