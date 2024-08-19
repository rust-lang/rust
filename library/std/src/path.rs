//! Cross-platform path manipulation.
//!
//! This module provides two types, [`PathBuf`] and [`Path`] (akin to [`String`]
//! and [`str`]), for working with paths abstractly. These types are thin wrappers
//! around [`OsString`] and [`OsStr`] respectively, meaning that they work directly
//! on strings according to the local platform's path syntax.
//!
//! Paths can be parsed into [`Component`]s by iterating over the structure
//! returned by the [`components`] method on [`Path`]. [`Component`]s roughly
//! correspond to the substrings between path separators (`/` or `\`). You can
//! reconstruct an equivalent path from components with the [`push`] method on
//! [`PathBuf`]; note that the paths may differ syntactically by the
//! normalization described in the documentation for the [`components`] method.
//!
//! ## Case sensitivity
//!
//! Unless otherwise indicated path methods that do not access the filesystem,
//! such as [`Path::starts_with`] and [`Path::ends_with`], are case sensitive no
//! matter the platform or filesystem. An exception to this is made for Windows
//! drive letters.
//!
//! ## Simple usage
//!
//! Path manipulation includes both parsing components from slices and building
//! new owned paths.
//!
//! To parse a path, you can create a [`Path`] slice from a [`str`]
//! slice and start asking questions:
//!
//! ```
//! use std::path::Path;
//! use std::ffi::OsStr;
//!
//! let path = Path::new("/tmp/foo/bar.txt");
//!
//! let parent = path.parent();
//! assert_eq!(parent, Some(Path::new("/tmp/foo")));
//!
//! let file_stem = path.file_stem();
//! assert_eq!(file_stem, Some(OsStr::new("bar")));
//!
//! let extension = path.extension();
//! assert_eq!(extension, Some(OsStr::new("txt")));
//! ```
//!
//! To build or modify paths, use [`PathBuf`]:
//!
//! ```
//! use std::path::PathBuf;
//!
//! // This way works...
//! let mut path = PathBuf::from("c:\\");
//!
//! path.push("windows");
//! path.push("system32");
//!
//! path.set_extension("dll");
//!
//! // ... but push is best used if you don't know everything up
//! // front. If you do, this way is better:
//! let path: PathBuf = ["c:\\", "windows", "system32.dll"].iter().collect();
//! ```
//!
//! [`components`]: Path::components
//! [`push`]: PathBuf::push

#![stable(feature = "rust1", since = "1.0.0")]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(test)]
mod tests;

#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc::path::PathBuf;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::path::{
    is_separator, Ancestors, Component, Components, Display, Iter, Path, Prefix, PrefixComponent,
    StripPrefixError, MAIN_SEPARATOR, MAIN_SEPARATOR_STR,
};

use crate::{fs, io, sys};

////////////////////////////////////////////////////////////////////////////////
// GENERAL NOTES
////////////////////////////////////////////////////////////////////////////////
//
// Parsing in this module is done by directly transmuting OsStr to [u8] slices,
// taking advantage of the fact that OsStr always encodes ASCII characters
// as-is.  Eventually, this transmutation should be replaced by direct uses of
// OsStr APIs for parsing, but it will take a while for those to become
// available.

impl Path {
    /// Queries the file system to get information about a file, directory, etc.
    ///
    /// This function will traverse symbolic links to query information about the
    /// destination file.
    ///
    /// This is an alias to [`fs::metadata`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    ///
    /// let path = Path::new("/Minas/tirith");
    /// let metadata = path.metadata().expect("metadata call failed");
    /// println!("{:?}", metadata.file_type());
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[inline]
    pub fn metadata(&self) -> io::Result<fs::Metadata> {
        fs::metadata(self)
    }

    /// Queries the metadata about a file without following symlinks.
    ///
    /// This is an alias to [`fs::symlink_metadata`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    ///
    /// let path = Path::new("/Minas/tirith");
    /// let metadata = path.symlink_metadata().expect("symlink_metadata call failed");
    /// println!("{:?}", metadata.file_type());
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[inline]
    pub fn symlink_metadata(&self) -> io::Result<fs::Metadata> {
        fs::symlink_metadata(self)
    }

    /// Returns the canonical, absolute form of the path with all intermediate
    /// components normalized and symbolic links resolved.
    ///
    /// This is an alias to [`fs::canonicalize`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::{Path, PathBuf};
    ///
    /// let path = Path::new("/foo/test/../test/bar.rs");
    /// assert_eq!(path.canonicalize().unwrap(), PathBuf::from("/foo/test/bar.rs"));
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[inline]
    pub fn canonicalize(&self) -> io::Result<PathBuf> {
        fs::canonicalize(self)
    }

    /// Reads a symbolic link, returning the file that the link points to.
    ///
    /// This is an alias to [`fs::read_link`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    ///
    /// let path = Path::new("/laputa/sky_castle.rs");
    /// let path_link = path.read_link().expect("read_link call failed");
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[inline]
    pub fn read_link(&self) -> io::Result<PathBuf> {
        fs::read_link(self)
    }

    /// Returns an iterator over the entries within a directory.
    ///
    /// The iterator will yield instances of <code>[io::Result]<[fs::DirEntry]></code>. New
    /// errors may be encountered after an iterator is initially constructed.
    ///
    /// This is an alias to [`fs::read_dir`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    ///
    /// let path = Path::new("/laputa");
    /// for entry in path.read_dir().expect("read_dir call failed") {
    ///     if let Ok(entry) = entry {
    ///         println!("{:?}", entry.path());
    ///     }
    /// }
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[inline]
    pub fn read_dir(&self) -> io::Result<fs::ReadDir> {
        fs::read_dir(self)
    }

    /// Returns `true` if the path points at an existing entity.
    ///
    /// Warning: this method may be error-prone, consider using [`try_exists()`] instead!
    /// It also has a risk of introducing time-of-check to time-of-use (TOCTOU) bugs.
    ///
    /// This function will traverse symbolic links to query information about the
    /// destination file.
    ///
    /// If you cannot access the metadata of the file, e.g. because of a
    /// permission error or broken symbolic links, this will return `false`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    /// assert!(!Path::new("does_not_exist.txt").exists());
    /// ```
    ///
    /// # See Also
    ///
    /// This is a convenience function that coerces errors to false. If you want to
    /// check errors, call [`Path::try_exists`].
    ///
    /// [`try_exists()`]: Self::try_exists
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[must_use]
    #[inline]
    pub fn exists(&self) -> bool {
        fs::metadata(self).is_ok()
    }

    /// Returns `Ok(true)` if the path points at an existing entity.
    ///
    /// This function will traverse symbolic links to query information about the
    /// destination file. In case of broken symbolic links this will return `Ok(false)`.
    ///
    /// [`Path::exists()`] only checks whether or not a path was both found and readable. By
    /// contrast, `try_exists` will return `Ok(true)` or `Ok(false)`, respectively, if the path
    /// was _verified_ to exist or not exist. If its existence can neither be confirmed nor
    /// denied, it will propagate an `Err(_)` instead. This can be the case if e.g. listing
    /// permission is denied on one of the parent directories.
    ///
    /// Note that while this avoids some pitfalls of the `exists()` method, it still can not
    /// prevent time-of-check to time-of-use (TOCTOU) bugs. You should only use it in scenarios
    /// where those bugs are not an issue.
    ///
    /// This is an alias for [`std::fs::exists`](crate::fs::exists).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    /// assert!(!Path::new("does_not_exist.txt").try_exists().expect("Can't check existence of file does_not_exist.txt"));
    /// assert!(Path::new("/root/secret_file.txt").try_exists().is_err());
    /// ```
    ///
    /// [`exists()`]: Self::exists
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_try_exists", since = "1.63.0")]
    #[inline]
    pub fn try_exists(&self) -> io::Result<bool> {
        fs::exists(self)
    }

    /// Returns `true` if the path exists on disk and is pointing at a regular file.
    ///
    /// This function will traverse symbolic links to query information about the
    /// destination file.
    ///
    /// If you cannot access the metadata of the file, e.g. because of a
    /// permission error or broken symbolic links, this will return `false`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    /// assert_eq!(Path::new("./is_a_directory/").is_file(), false);
    /// assert_eq!(Path::new("a_file.txt").is_file(), true);
    /// ```
    ///
    /// # See Also
    ///
    /// This is a convenience function that coerces errors to false. If you want to
    /// check errors, call [`fs::metadata`] and handle its [`Result`]. Then call
    /// [`fs::Metadata::is_file`] if it was [`Ok`].
    ///
    /// When the goal is simply to read from (or write to) the source, the most
    /// reliable way to test the source can be read (or written to) is to open
    /// it. Only using `is_file` can break workflows like `diff <( prog_a )` on
    /// a Unix-like system for example. See [`fs::File::open`] or
    /// [`fs::OpenOptions::open`] for more information.
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[must_use]
    pub fn is_file(&self) -> bool {
        fs::metadata(self).map(|m| m.is_file()).unwrap_or(false)
    }

    /// Returns `true` if the path exists on disk and is pointing at a directory.
    ///
    /// This function will traverse symbolic links to query information about the
    /// destination file.
    ///
    /// If you cannot access the metadata of the file, e.g. because of a
    /// permission error or broken symbolic links, this will return `false`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    /// assert_eq!(Path::new("./is_a_directory/").is_dir(), true);
    /// assert_eq!(Path::new("a_file.txt").is_dir(), false);
    /// ```
    ///
    /// # See Also
    ///
    /// This is a convenience function that coerces errors to false. If you want to
    /// check errors, call [`fs::metadata`] and handle its [`Result`]. Then call
    /// [`fs::Metadata::is_dir`] if it was [`Ok`].
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[must_use]
    pub fn is_dir(&self) -> bool {
        fs::metadata(self).map(|m| m.is_dir()).unwrap_or(false)
    }

    /// Returns `true` if the path exists on disk and is pointing at a symbolic link.
    ///
    /// This function will not traverse symbolic links.
    /// In case of a broken symbolic link this will also return true.
    ///
    /// If you cannot access the directory containing the file, e.g., because of a
    /// permission error, this will return false.
    ///
    /// # Examples
    ///
    #[rustc_allow_incoherent_impl]
    #[cfg_attr(unix, doc = "```no_run")]
    #[cfg_attr(not(unix), doc = "```ignore")]
    /// use std::path::Path;
    /// use std::os::unix::fs::symlink;
    ///
    /// let link_path = Path::new("link");
    /// symlink("/origin_does_not_exist/", link_path).unwrap();
    /// assert_eq!(link_path.is_symlink(), true);
    /// assert_eq!(link_path.exists(), false);
    /// ```
    ///
    /// # See Also
    ///
    /// This is a convenience function that coerces errors to false. If you want to
    /// check errors, call [`fs::symlink_metadata`] and handle its [`Result`]. Then call
    /// [`fs::Metadata::is_symlink`] if it was [`Ok`].
    #[must_use]
    #[stable(feature = "is_symlink", since = "1.58.0")]
    pub fn is_symlink(&self) -> bool {
        fs::symlink_metadata(self).map(|m| m.is_symlink()).unwrap_or(false)
    }
}

/// Makes the path absolute without accessing the filesystem.
///
/// If the path is relative, the current directory is used as the base directory.
/// All intermediate components will be resolved according to platform-specific
/// rules, but unlike [`canonicalize`][crate::fs::canonicalize], this does not
/// resolve symlinks and may succeed even if the path does not exist.
///
/// If the `path` is empty or getting the
/// [current directory][crate::env::current_dir] fails, then an error will be
/// returned.
///
/// # Platform-specific behavior
///
/// On POSIX platforms, the path is resolved using [POSIX semantics][posix-semantics],
/// except that it stops short of resolving symlinks. This means it will keep `..`
/// components and trailing slashes.
///
/// On Windows, for verbatim paths, this will simply return the path as given. For other
/// paths, this is currently equivalent to calling
/// [`GetFullPathNameW`][windows-path].
///
/// Note that these [may change in the future][changes].
///
/// # Errors
///
/// This function may return an error in the following situations:
///
/// * If `path` is syntactically invalid; in particular, if it is empty.
/// * If getting the [current directory][crate::env::current_dir] fails.
///
/// # Examples
///
/// ## POSIX paths
///
/// ```
/// # #[cfg(unix)]
/// fn main() -> std::io::Result<()> {
///     use std::path::{self, Path};
///
///     // Relative to absolute
///     let absolute = path::absolute("foo/./bar")?;
///     assert!(absolute.ends_with("foo/bar"));
///
///     // Absolute to absolute
///     let absolute = path::absolute("/foo//test/.././bar.rs")?;
///     assert_eq!(absolute, Path::new("/foo/test/../bar.rs"));
///     Ok(())
/// }
/// # #[cfg(not(unix))]
/// # fn main() {}
/// ```
///
/// ## Windows paths
///
/// ```
/// # #[cfg(windows)]
/// fn main() -> std::io::Result<()> {
///     use std::path::{self, Path};
///
///     // Relative to absolute
///     let absolute = path::absolute("foo/./bar")?;
///     assert!(absolute.ends_with(r"foo\bar"));
///
///     // Absolute to absolute
///     let absolute = path::absolute(r"C:\foo//test\..\./bar.rs")?;
///
///     assert_eq!(absolute, Path::new(r"C:\foo\bar.rs"));
///     Ok(())
/// }
/// # #[cfg(not(windows))]
/// # fn main() {}
/// ```
///
/// Note that this [may change in the future][changes].
///
/// [changes]: io#platform-specific-behavior
/// [posix-semantics]: https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_13
/// [windows-path]: https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfullpathnamew
#[stable(feature = "absolute_path", since = "1.79.0")]
pub fn absolute<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    let path = path.as_ref();
    if path.as_os_str().is_empty() {
        Err(io::const_io_error!(io::ErrorKind::InvalidInput, "cannot make an empty path absolute",))
    } else {
        sys::path::absolute(path)
    }
}
