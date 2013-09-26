// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Cross-platform file path handling (re-write)

use container::Container;
use c_str::CString;
use clone::Clone;
use iter::Iterator;
use option::{Option, None, Some};
use str;
use str::{OwnedStr, Str, StrSlice};
use vec;
use vec::{CopyableVector, OwnedCopyableVector, OwnedVector};
use vec::{ImmutableEqVector, ImmutableVector};

pub mod posix;
pub mod windows;

/// Typedef for POSIX file paths.
/// See `posix::Path` for more info.
pub type PosixPath = posix::Path;

/// Typedef for Windows file paths.
/// See `windows::Path` for more info.
pub type WindowsPath = windows::Path;

/// Typedef for the platform-native path type
#[cfg(unix)]
pub type Path = PosixPath;
/// Typedef for the platform-native path type
#[cfg(windows)]
pub type Path = WindowsPath;

/// Typedef for the POSIX path component iterator.
/// See `posix::ComponentIter` for more info.
pub type PosixComponentIter<'self> = posix::ComponentIter<'self>;

// /// Typedef for the Windows path component iterator.
// /// See `windows::ComponentIter` for more info.
// pub type WindowsComponentIter<'self> = windows::ComponentIter<'self>;

/// Typedef for the platform-native component iterator
#[cfg(unix)]
pub type ComponentIter<'self> = PosixComponentIter<'self>;
// /// Typedef for the platform-native component iterator
//#[cfg(windows)]
//pub type ComponentIter<'self> = WindowsComponentIter<'self>;

// Condition that is raised when a NUL is found in a byte vector given to a Path function
condition! {
    // this should be a &[u8] but there's a lifetime issue
    null_byte: ~[u8] -> ~[u8];
}

/// A trait that represents the generic operations available on paths
pub trait GenericPath: Clone + GenericPathUnsafe {
    /// Creates a new Path from a byte vector.
    /// The resulting Path will always be normalized.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the path contains a NUL.
    ///
    /// See individual Path impls for additional restrictions.
    #[inline]
    fn from_vec(path: &[u8]) -> Self {
        if contains_nul(path) {
            let path = self::null_byte::cond.raise(path.to_owned());
            assert!(!contains_nul(path));
            unsafe { GenericPathUnsafe::from_vec_unchecked(path) }
        } else {
            unsafe { GenericPathUnsafe::from_vec_unchecked(path) }
        }
    }

    /// Creates a new Path from a byte vector, if possible.
    /// The resulting Path will always be normalized.
    #[inline]
    fn from_vec_opt(path: &[u8]) -> Option<Self> {
        if contains_nul(path) {
            None
        } else {
            Some(unsafe { GenericPathUnsafe::from_vec_unchecked(path) })
        }
    }

    /// Creates a new Path from a string.
    /// The resulting Path will always be normalized.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the path contains a NUL.
    #[inline]
    fn from_str(path: &str) -> Self {
        let v = path.as_bytes();
        if contains_nul(v) {
            GenericPath::from_vec(path.as_bytes()) // let from_vec handle the condition
        } else {
            unsafe { GenericPathUnsafe::from_str_unchecked(path) }
        }
    }

    /// Creates a new Path from a string, if possible.
    /// The resulting Path will always be normalized.
    #[inline]
    fn from_str_opt(path: &str) -> Option<Self> {
        let v = path.as_bytes();
        if contains_nul(v) {
            None
        } else {
            Some(unsafe { GenericPathUnsafe::from_str_unchecked(path) })
        }
    }

    /// Creates a new Path from a CString.
    /// The resulting Path will always be normalized.
    ///
    /// See individual Path impls for potential restrictions.
    #[inline]
    fn from_c_str(path: CString) -> Self {
        // CStrings can't contain NULs
        unsafe { GenericPathUnsafe::from_vec_unchecked(path.as_bytes()) }
    }

    /// Returns the path as a string, if possible.
    /// If the path is not representable in utf-8, this returns None.
    #[inline]
    fn as_str<'a>(&'a self) -> Option<&'a str> {
        str::from_utf8_slice_opt(self.as_vec())
    }

    /// Returns the path as a byte vector
    fn as_vec<'a>(&'a self) -> &'a [u8];

    /// Provides the path as a string
    ///
    /// If the path is not UTF-8, invalid sequences will be replaced with the unicode
    /// replacement char. This involves allocation.
    #[inline]
    fn as_display_str<T>(&self, f: &fn(&str) -> T) -> T {
        match self.as_str() {
            Some(s) => f(s),
            None => {
                let s = self.to_display_str();
                f(s.as_slice())
            }
        }
    }

    /// Returns the path as a string
    ///
    /// If the path is not UTF-8, invalid sequences will be replaced with the unicode
    /// replacement char. This involves allocation.
    ///
    /// This is similar to `as_display_str()` except it will always allocate a new ~str.
    fn to_display_str(&self) -> ~str {
        // FIXME (#9516): Don't decode utf-8 manually here once we have a good way to do it in str
        // This is a truly horrifically bad implementation, done as a functionality stopgap until
        // we have a proper utf-8 decoder. I don't really want to write one here.
        static REPLACEMENT_CHAR: char = '\uFFFD';

        let mut v = self.as_vec();
        let mut s = str::with_capacity(v.len());
        while !v.is_empty() {
            let w = str::utf8_char_width(v[0]);
            if w == 0u {
                s.push_char(REPLACEMENT_CHAR);
                v = v.slice_from(1);
            } else if v.len() < w || !str::is_utf8(v.slice_to(w)) {
                s.push_char(REPLACEMENT_CHAR);
                v = v.slice_from(1);
            } else {
                s.push_str(unsafe { ::cast::transmute(v.slice_to(w)) });
                v = v.slice_from(w);
            }
        }
        s
    }

    /// Returns the directory component of `self`, as a byte vector (with no trailing separator).
    /// If `self` has no directory component, returns ['.'].
    fn dirname<'a>(&'a self) -> &'a [u8];
    /// Returns the directory component of `self`, as a string, if possible.
    /// See `dirname` for details.
    #[inline]
    fn dirname_str<'a>(&'a self) -> Option<&'a str> {
        str::from_utf8_slice_opt(self.dirname())
    }
    /// Returns the file component of `self`, as a byte vector.
    /// If `self` represents the root of the file hierarchy, returns the empty vector.
    /// If `self` is ".", returns the empty vector.
    fn filename<'a>(&'a self) -> &'a [u8];
    /// Returns the file component of `self`, as a string, if possible.
    /// See `filename` for details.
    #[inline]
    fn filename_str<'a>(&'a self) -> Option<&'a str> {
        str::from_utf8_slice_opt(self.filename())
    }
    /// Returns the stem of the filename of `self`, as a byte vector.
    /// The stem is the portion of the filename just before the last '.'.
    /// If there is no '.', the entire filename is returned.
    fn filestem<'a>(&'a self) -> &'a [u8] {
        let name = self.filename();
        let dot = '.' as u8;
        match name.rposition_elem(&dot) {
            None | Some(0) => name,
            Some(1) if name == bytes!("..") => name,
            Some(pos) => name.slice_to(pos)
        }
    }
    /// Returns the stem of the filename of `self`, as a string, if possible.
    /// See `filestem` for details.
    #[inline]
    fn filestem_str<'a>(&'a self) -> Option<&'a str> {
        str::from_utf8_slice_opt(self.filestem())
    }
    /// Returns the extension of the filename of `self`, as an optional byte vector.
    /// The extension is the portion of the filename just after the last '.'.
    /// If there is no extension, None is returned.
    /// If the filename ends in '.', the empty vector is returned.
    fn extension<'a>(&'a self) -> Option<&'a [u8]> {
        let name = self.filename();
        let dot = '.' as u8;
        match name.rposition_elem(&dot) {
            None | Some(0) => None,
            Some(1) if name == bytes!("..") => None,
            Some(pos) => Some(name.slice_from(pos+1))
        }
    }
    /// Returns the extension of the filename of `self`, as a string, if possible.
    /// See `extension` for details.
    #[inline]
    fn extension_str<'a>(&'a self) -> Option<&'a str> {
        self.extension().and_then(|v| str::from_utf8_slice_opt(v))
    }

    /// Replaces the directory portion of the path with the given byte vector.
    /// If `self` represents the root of the filesystem hierarchy, the last path component
    /// of the given byte vector becomes the filename.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the dirname contains a NUL.
    #[inline]
    fn set_dirname(&mut self, dirname: &[u8]) {
        if contains_nul(dirname) {
            let dirname = self::null_byte::cond.raise(dirname.to_owned());
            assert!(!contains_nul(dirname));
            unsafe { self.set_dirname_unchecked(dirname) }
        } else {
            unsafe { self.set_dirname_unchecked(dirname) }
        }
    }
    /// Replaces the directory portion of the path with the given string.
    /// See `set_dirname` for details.
    #[inline]
    fn set_dirname_str(&mut self, dirname: &str) {
        if contains_nul(dirname.as_bytes()) {
            self.set_dirname(dirname.as_bytes()) // triggers null_byte condition
        } else {
            unsafe { self.set_dirname_str_unchecked(dirname) }
        }
    }
    /// Replaces the filename portion of the path with the given byte vector.
    /// If the replacement name is [], this is equivalent to popping the path.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the filename contains a NUL.
    #[inline]
    fn set_filename(&mut self, filename: &[u8]) {
        if contains_nul(filename) {
            let filename = self::null_byte::cond.raise(filename.to_owned());
            assert!(!contains_nul(filename));
            unsafe { self.set_filename_unchecked(filename) }
        } else {
            unsafe { self.set_filename_unchecked(filename) }
        }
    }
    /// Replaces the filename portion of the path with the given string.
    /// See `set_filename` for details.
    #[inline]
    fn set_filename_str(&mut self, filename: &str) {
        if contains_nul(filename.as_bytes()) {
            self.set_filename(filename.as_bytes()) // triggers null_byte condition
        } else {
            unsafe { self.set_filename_str_unchecked(filename) }
        }
    }
    /// Replaces the filestem with the given byte vector.
    /// If there is no extension in `self` (or `self` has no filename), this is equivalent
    /// to `set_filename`. Otherwise, if the given byte vector is [], the extension (including
    /// the preceding '.') becomes the new filename.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the filestem contains a NUL.
    fn set_filestem(&mut self, filestem: &[u8]) {
        // borrowck is being a pain here
        let val = {
            let name = self.filename();
            if !name.is_empty() {
                let dot = '.' as u8;
                match name.rposition_elem(&dot) {
                    None | Some(0) => None,
                    Some(idx) => {
                        let mut v;
                        if contains_nul(filestem) {
                            let filestem = self::null_byte::cond.raise(filestem.to_owned());
                            assert!(!contains_nul(filestem));
                            v = vec::with_capacity(filestem.len() + name.len() - idx);
                            v.push_all(filestem);
                        } else {
                            v = vec::with_capacity(filestem.len() + name.len() - idx);
                            v.push_all(filestem);
                        }
                        v.push_all(name.slice_from(idx));
                        Some(v)
                    }
                }
            } else { None }
        };
        match val {
            None => self.set_filename(filestem),
            Some(v) => unsafe { self.set_filename_unchecked(v) }
        }
    }
    /// Replaces the filestem with the given string.
    /// See `set_filestem` for details.
    #[inline]
    fn set_filestem_str(&mut self, filestem: &str) {
        self.set_filestem(filestem.as_bytes())
    }
    /// Replaces the extension with the given byte vector.
    /// If there is no extension in `self`, this adds one.
    /// If the given byte vector is [], this removes the extension.
    /// If `self` has no filename, this is a no-op.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the extension contains a NUL.
    fn set_extension(&mut self, extension: &[u8]) {
        // borrowck causes problems here too
        let val = {
            let name = self.filename();
            if !name.is_empty() {
                let dot = '.' as u8;
                match name.rposition_elem(&dot) {
                    None | Some(0) => {
                        if extension.is_empty() {
                            None
                        } else {
                            let mut v;
                            if contains_nul(extension) {
                                let extension = self::null_byte::cond.raise(extension.to_owned());
                                assert!(!contains_nul(extension));
                                v = vec::with_capacity(name.len() + extension.len() + 1);
                                v.push_all(name);
                                v.push(dot);
                                v.push_all(extension);
                            } else {
                                v = vec::with_capacity(name.len() + extension.len() + 1);
                                v.push_all(name);
                                v.push(dot);
                                v.push_all(extension);
                            }
                            Some(v)
                        }
                    }
                    Some(idx) => {
                        if extension.is_empty() {
                            Some(name.slice_to(idx).to_owned())
                        } else {
                            let mut v;
                            if contains_nul(extension) {
                                let extension = self::null_byte::cond.raise(extension.to_owned());
                                assert!(!contains_nul(extension));
                                v = vec::with_capacity(idx + extension.len() + 1);
                                v.push_all(name.slice_to(idx+1));
                                v.push_all(extension);
                            } else {
                                v = vec::with_capacity(idx + extension.len() + 1);
                                v.push_all(name.slice_to(idx+1));
                                v.push_all(extension);
                            }
                            Some(v)
                        }
                    }
                }
            } else { None }
        };
        match val {
            None => (),
            Some(v) => unsafe { self.set_filename_unchecked(v) }
        }
    }
    /// Replaces the extension with the given string.
    /// See `set_extension` for details.
    #[inline]
    fn set_extension_str(&mut self, extension: &str) {
        self.set_extension(extension.as_bytes())
    }

    /// Returns a new Path constructed by replacing the dirname with the given byte vector.
    /// See `set_dirname` for details.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the dirname contains a NUL.
    #[inline]
    fn with_dirname(&self, dirname: &[u8]) -> Self {
        let mut p = self.clone();
        p.set_dirname(dirname);
        p
    }
    /// Returns a new Path constructed by replacing the dirname with the given string.
    /// See `set_dirname` for details.
    #[inline]
    fn with_dirname_str(&self, dirname: &str) -> Self {
        let mut p = self.clone();
        p.set_dirname_str(dirname);
        p
    }
    /// Returns a new Path constructed by replacing the filename with the given byte vector.
    /// See `set_filename` for details.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the filename contains a NUL.
    #[inline]
    fn with_filename(&self, filename: &[u8]) -> Self {
        let mut p = self.clone();
        p.set_filename(filename);
        p
    }
    /// Returns a new Path constructed by replacing the filename with the given string.
    /// See `set_filename` for details.
    #[inline]
    fn with_filename_str(&self, filename: &str) -> Self {
        let mut p = self.clone();
        p.set_filename_str(filename);
        p
    }
    /// Returns a new Path constructed by setting the filestem to the given byte vector.
    /// See `set_filestem` for details.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the filestem contains a NUL.
    #[inline]
    fn with_filestem(&self, filestem: &[u8]) -> Self {
        let mut p = self.clone();
        p.set_filestem(filestem);
        p
    }
    /// Returns a new Path constructed by setting the filestem to the given string.
    /// See `set_filestem` for details.
    #[inline]
    fn with_filestem_str(&self, filestem: &str) -> Self {
        let mut p = self.clone();
        p.set_filestem_str(filestem);
        p
    }
    /// Returns a new Path constructed by setting the extension to the given byte vector.
    /// See `set_extension` for details.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the extension contains a NUL.
    #[inline]
    fn with_extension(&self, extension: &[u8]) -> Self {
        let mut p = self.clone();
        p.set_extension(extension);
        p
    }
    /// Returns a new Path constructed by setting the extension to the given string.
    /// See `set_extension` for details.
    #[inline]
    fn with_extension_str(&self, extension: &str) -> Self {
        let mut p = self.clone();
        p.set_extension_str(extension);
        p
    }

    /// Returns the directory component of `self`, as a Path.
    /// If `self` represents the root of the filesystem hierarchy, returns `self`.
    fn dir_path(&self) -> Self {
        // self.dirname() returns a NUL-free vector
        unsafe { GenericPathUnsafe::from_vec_unchecked(self.dirname()) }
    }
    /// Returns the file component of `self`, as a relative Path.
    /// If `self` represents the root of the filesystem hierarchy, returns None.
    fn file_path(&self) -> Option<Self> {
        // self.filename() returns a NUL-free vector
        match self.filename() {
            [] => None,
            v => Some(unsafe { GenericPathUnsafe::from_vec_unchecked(v) })
        }
    }

    /// Pushes a path (as a byte vector) onto `self`.
    /// If the argument represents an absolute path, it replaces `self`.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the path contains a NUL.
    #[inline]
    fn push(&mut self, path: &[u8]) {
        if contains_nul(path) {
            let path = self::null_byte::cond.raise(path.to_owned());
            assert!(!contains_nul(path));
            unsafe { self.push_unchecked(path) }
        } else {
            unsafe { self.push_unchecked(path) }
        }
    }
    /// Pushes a path (as a string) onto `self.
    /// See `push` for details.
    #[inline]
    fn push_str(&mut self, path: &str) {
        if contains_nul(path.as_bytes()) {
            self.push(path.as_bytes()) // triggers null_byte condition
        } else {
            unsafe { self.push_str_unchecked(path) }
        }
    }
    /// Pushes a Path onto `self`.
    /// If the argument represents an absolute path, it replaces `self`.
    #[inline]
    fn push_path(&mut self, path: &Self) {
        self.push(path.as_vec())
    }
    /// Pops the last path component off of `self` and returns it.
    /// If `self` represents the root of the file hierarchy, None is returned.
    fn pop(&mut self) -> Option<~[u8]>;
    /// Pops the last path component off of `self` and returns it as a string, if possible.
    /// `self` will still be modified even if None is returned.
    /// See `pop` for details.
    #[inline]
    fn pop_str(&mut self) -> Option<~str> {
        self.pop().and_then(|v| str::from_utf8_owned_opt(v))
    }

    /// Returns a new Path constructed by joining `self` with the given path (as a byte vector).
    /// If the given path is absolute, the new Path will represent just that.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the path contains a NUL.
    #[inline]
    fn join(&self, path: &[u8]) -> Self {
        let mut p = self.clone();
        p.push(path);
        p
    }
    /// Returns a new Path constructed by joining `self` with the given path (as a string).
    /// See `join` for details.
    #[inline]
    fn join_str(&self, path: &str) -> Self {
        let mut p = self.clone();
        p.push_str(path);
        p
    }
    /// Returns a new Path constructed by joining `self` with the given path.
    /// If the given path is absolute, the new Path will represent just that.
    #[inline]
    fn join_path(&self, path: &Self) -> Self {
        let mut p = self.clone();
        p.push_path(path);
        p
    }

    /// Returns whether `self` represents an absolute path.
    /// An absolute path is defined as one that, when joined to another path, will
    /// yield back the same absolute path.
    fn is_absolute(&self) -> bool;

    /// Returns whether `self` is equal to, or is an ancestor of, the given path.
    /// If both paths are relative, they are compared as though they are relative
    /// to the same parent path.
    fn is_ancestor_of(&self, other: &Self) -> bool;

    /// Returns the Path that, were it joined to `base`, would yield `self`.
    /// If no such path exists, None is returned.
    /// If `self` is absolute and `base` is relative, or on Windows if both
    /// paths refer to separate drives, an absolute path is returned.
    fn path_relative_from(&self, base: &Self) -> Option<Self>;

    /// Executes a callback with the receiver and every parent
    fn each_parent(&self, f: &fn(&Self) -> bool) -> bool {
        let mut p = self.clone();
        loop {
            if !f(&p) {
                return false;
            }
            let f = p.pop();
            if f.is_none() || bytes!("..") == f.unwrap() {
                break;
            }
        }
        true
    }
}

/// A trait that represents the unsafe operations on GenericPaths
pub trait GenericPathUnsafe {
    /// Creates a new Path from a byte vector without checking for null bytes.
    /// The resulting Path will always be normalized.
    unsafe fn from_vec_unchecked(path: &[u8]) -> Self;

    /// Creates a new Path from a str without checking for null bytes.
    /// The resulting Path will always be normalized.
    #[inline]
    unsafe fn from_str_unchecked(path: &str) -> Self {
        GenericPathUnsafe::from_vec_unchecked(path.as_bytes())
    }

    /// Replaces the directory portion of the path with the given byte vector without
    /// checking for null bytes.
    /// See `set_dirname` for details.
    unsafe fn set_dirname_unchecked(&mut self, dirname: &[u8]);

    /// Replaces the directory portion of the path with the given str without
    /// checking for null bytes.
    /// See `set_dirname_str` for details.
    #[inline]
    unsafe fn set_dirname_str_unchecked(&mut self, dirname: &str) {
        self.set_dirname_unchecked(dirname.as_bytes())
    }

    /// Replaces the filename portion of the path with the given byte vector without
    /// checking for null bytes.
    /// See `set_filename` for details.
    unsafe fn set_filename_unchecked(&mut self, filename: &[u8]);

    /// Replaces the filename portion of the path with the given str without
    /// checking for null bytes.
    /// See `set_filename_str` for details.
    #[inline]
    unsafe fn set_filename_str_unchecked(&mut self, filename: &str) {
        self.set_filename_unchecked(filename.as_bytes())
    }

    /// Pushes a byte vector onto `self` without checking for null bytes.
    /// See `push` for details.
    unsafe fn push_unchecked(&mut self, path: &[u8]);

    /// Pushes a str onto `self` without checking for null bytes.
    /// See `push_str` for details.
    #[inline]
    unsafe fn push_str_unchecked(&mut self, path: &str) {
        self.push_unchecked(path.as_bytes())
    }
}

#[inline(always)]
fn contains_nul(v: &[u8]) -> bool {
    v.iter().any(|&x| x == 0)
}

// FIXME (#9537): libc::stat should derive Default
#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
mod stat {
    #[allow(missing_doc)];

    #[cfg(target_arch = "x86")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                __pad1: 0,
                st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                __pad2: 0,
                st_size: 0,
                st_blksize: 0,
                st_blocks: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                __unused4: 0,
                __unused5: 0,
            }
        }
    }

    #[cfg(target_arch = "arm")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                __pad0: [0, ..4],
                __st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                __pad3: [0, ..4],
                st_size: 0,
                st_blksize: 0,
                st_blocks: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                st_ino: 0
            }
        }
    }

    #[cfg(target_arch = "mips")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_pad1: [0, ..3],
                st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                st_pad2: [0, ..2],
                st_size: 0,
                st_pad3: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                st_blksize: 0,
                st_blocks: 0,
                st_pad5: [0, ..14],
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_ino: 0,
                st_nlink: 0,
                st_mode: 0,
                st_uid: 0,
                st_gid: 0,
                __pad0: 0,
                st_rdev: 0,
                st_size: 0,
                st_blksize: 0,
                st_blocks: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                __unused: [0, 0, 0],
            }
        }
    }
}

#[cfg(target_os = "freebsd")]
mod stat {
    #[allow(missing_doc)];

    #[cfg(target_arch = "x86_64")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                st_size: 0,
                st_blocks: 0,
                st_blksize: 0,
                st_flags: 0,
                st_gen: 0,
                st_lspare: 0,
                st_birthtime: 0,
                st_birthtime_nsec: 0,
                __unused: [0, 0],
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod stat {
    #[allow(missing_doc)];

    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_mode: 0,
                st_nlink: 0,
                st_ino: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                st_birthtime: 0,
                st_birthtime_nsec: 0,
                st_size: 0,
                st_blocks: 0,
                st_blksize: 0,
                st_flags: 0,
                st_gen: 0,
                st_lspare: 0,
                st_qspare: [0, 0],
            }
        }
    }
}

#[cfg(target_os = "win32")]
mod stat {
    #[allow(missing_doc)];

    pub mod arch {
        use libc;
        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                st_size: 0,
                st_atime: 0,
                st_mtime: 0,
                st_ctime: 0,
            }
        }
    }
}
