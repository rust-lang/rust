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
use c_str::{CString, ToCStr};
use clone::Clone;
use cmp::Eq;
use from_str::FromStr;
use iterator::{AdditiveIterator, Extendable, Iterator};
use option::{Option, None, Some};
use str;
use str::{OwnedStr, Str, StrSlice};
use util;
use vec;
use vec::{CopyableVector, OwnedCopyableVector, OwnedVector};
use vec::{ImmutableEqVector, ImmutableVector, Vector, VectorVector};

/// Typedef for the platform-native path type
#[cfg(unix)]
pub type Path = PosixPath;
// /// Typedef for the platform-native path type
//#[cfg(windows)]
//pub type Path = WindowsPath;

/// Typedef for the platform-native component iterator
#[cfg(unix)]
pub type ComponentIter<'self> = PosixComponentIter<'self>;
// /// Typedef for the platform-native component iterator
//#[cfg(windows)]
//pub type ComponentIter<'self> = WindowsComponentIter<'self>;

/// Iterator that yields successive components of a PosixPath
type PosixComponentIter<'self> = vec::SplitIterator<'self, u8>;

// Condition that is raised when a NUL is found in a byte vector given to a Path function
condition! {
    // this should be a &[u8] but there's a lifetime issue
    null_byte: ~[u8] -> ~[u8];
}

/// Represents a POSIX file path
#[deriving(Clone, DeepClone)]
pub struct PosixPath {
    priv repr: ~[u8], // assumed to never be empty or contain NULs
    priv sepidx: Option<uint> // index of the final separator in repr
}

impl Eq for PosixPath {
    fn eq(&self, other: &PosixPath) -> bool {
        self.repr == other.repr
    }
}

impl FromStr for PosixPath {
    fn from_str(s: &str) -> Option<PosixPath> {
        let v = s.as_bytes();
        if contains_nul(v) {
            None
        } else {
            Some(unsafe { GenericPathUnsafe::from_vec_unchecked(v) })
        }
    }
}

/// A trait that represents the generic operations available on paths
pub trait GenericPath: Clone + GenericPathUnsafe {
    /// Creates a new Path from a byte vector.
    /// The resulting Path will always be normalized.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the path contains a NUL.
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

    /// Creates a new Path from a string.
    /// The resulting Path will always be normalized.
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the path contains a NUL.
    #[inline]
    fn from_str(path: &str) -> Self {
        GenericPath::from_vec(path.as_bytes())
    }

    /// Creates a new Path from a CString.
    /// The resulting Path will always be normalized.
    #[inline]
    fn from_c_str(path: CString) -> Self {
        // CStrings can't contain NULs
        unsafe { GenericPathUnsafe::from_vec_unchecked(path.as_bytes()) }
    }

    /// Returns the path as a string, if possible.
    /// If the path is not representable in utf-8, this returns None.
    #[inline]
    fn as_str<'a>(&'a self) -> Option<&'a str> {
        str::from_bytes_slice_opt(self.as_vec())
    }

    /// Returns the path as a byte vector
    fn as_vec<'a>(&'a self) -> &'a [u8];

    /// Returns the directory component of `self`, as a byte vector (with no trailing separator).
    /// If `self` has no directory component, returns ['.'].
    fn dirname<'a>(&'a self) -> &'a [u8];
    /// Returns the directory component of `self`, as a string, if possible.
    /// See `dirname` for details.
    #[inline]
    fn dirname_str<'a>(&'a self) -> Option<&'a str> {
        str::from_bytes_slice_opt(self.dirname())
    }
    /// Returns the file component of `self`, as a byte vector.
    /// If `self` represents the root of the file hierarchy, returns the empty vector.
    /// If `self` is ".", returns the empty vector.
    fn filename<'a>(&'a self) -> &'a [u8];
    /// Returns the file component of `self`, as a string, if possible.
    /// See `filename` for details.
    #[inline]
    fn filename_str<'a>(&'a self) -> Option<&'a str> {
        str::from_bytes_slice_opt(self.filename())
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
        str::from_bytes_slice_opt(self.filestem())
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
        self.extension().chain(|v| str::from_bytes_slice_opt(v))
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
        self.set_dirname(dirname.as_bytes())
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
        self.set_filename(filename.as_bytes())
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
        self.with_dirname(dirname.as_bytes())
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
        self.with_filename(filename.as_bytes())
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
        self.with_filestem(filestem.as_bytes())
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
        self.with_extension(extension.as_bytes())
    }

    /// Returns the directory component of `self`, as a Path.
    /// If `self` represents the root of the filesystem hierarchy, returns `self`.
    fn dir_path(&self) -> Self {
        GenericPath::from_vec(self.dirname())
    }
    /// Returns the file component of `self`, as a relative Path.
    /// If `self` represents the root of the filesystem hierarchy, returns None.
    fn file_path(&self) -> Option<Self> {
        match self.filename() {
            [] => None,
            v => Some(GenericPath::from_vec(v))
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
        self.push(path.as_bytes())
    }
    /// Pushes a Path onto `self`.
    /// If the argument represents an absolute path, it replaces `self`.
    #[inline]
    fn push_path(&mut self, path: &Self) {
        self.push(path.as_vec())
    }
    /// Pops the last path component off of `self` and returns it.
    /// If `self` represents the root of the file hierarchy, None is returned.
    fn pop_opt(&mut self) -> Option<~[u8]>;
    /// Pops the last path component off of `self` and returns it as a string, if possible.
    /// `self` will still be modified even if None is returned.
    /// See `pop_opt` for details.
    #[inline]
    fn pop_opt_str(&mut self) -> Option<~str> {
        self.pop_opt().chain(|v| str::from_bytes_owned_opt(v))
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
        self.join(path.as_bytes())
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
}

/// A trait that represents the unsafe operations on GenericPaths
pub trait GenericPathUnsafe {
    /// Creates a new Path from a byte vector without checking for null bytes.
    /// The resulting Path will always be normalized.
    unsafe fn from_vec_unchecked(path: &[u8]) -> Self;

    /// Replaces the directory portion of the path with the given byte vector without
    /// checking for null bytes.
    /// See `set_dirname` for details.
    unsafe fn set_dirname_unchecked(&mut self, dirname: &[u8]);

    /// Replaces the filename portion of the path with the given byte vector without
    /// checking for null bytes.
    /// See `set_filename` for details.
    unsafe fn set_filename_unchecked(&mut self, filename: &[u8]);

    /// Pushes a path onto `self` without checking for null bytes.
    /// See `push` for details.
    unsafe fn push_unchecked(&mut self, path: &[u8]);
}

#[inline(always)]
fn contains_nul(v: &[u8]) -> bool {
    v.iter().any(|&x| x == 0)
}

impl ToCStr for PosixPath {
    #[inline]
    fn to_c_str(&self) -> CString {
        // The Path impl guarantees no internal NUL
        unsafe { self.as_vec().to_c_str_unchecked() }
    }

    #[inline]
    unsafe fn to_c_str_unchecked(&self) -> CString {
        self.as_vec().to_c_str_unchecked()
    }
}

impl GenericPathUnsafe for PosixPath {
    unsafe fn from_vec_unchecked(path: &[u8]) -> PosixPath {
        let path = PosixPath::normalize(path);
        assert!(!path.is_empty());
        let idx = path.rposition_elem(&posix::sep);
        PosixPath{ repr: path, sepidx: idx }
    }

    unsafe fn set_dirname_unchecked(&mut self, dirname: &[u8]) {
        match self.sepidx {
            None if bytes!(".") == self.repr || bytes!("..") == self.repr => {
                self.repr = PosixPath::normalize(dirname);
            }
            None => {
                let mut v = vec::with_capacity(dirname.len() + self.repr.len() + 1);
                v.push_all(dirname);
                v.push(posix::sep);
                v.push_all(self.repr);
                self.repr = PosixPath::normalize(v);
            }
            Some(0) if self.repr.len() == 1 && self.repr[0] == posix::sep => {
                self.repr = PosixPath::normalize(dirname);
            }
            Some(idx) if dirname.is_empty() => {
                let v = PosixPath::normalize(self.repr.slice_from(idx+1));
                self.repr = v;
            }
            Some(idx) if self.repr.slice_from(idx+1) == bytes!("..") => {
                self.repr = PosixPath::normalize(dirname);
            }
            Some(idx) => {
                let mut v = vec::with_capacity(dirname.len() + self.repr.len() - idx);
                v.push_all(dirname);
                v.push_all(self.repr.slice_from(idx));
                self.repr = PosixPath::normalize(v);
            }
        }
        self.sepidx = self.repr.rposition_elem(&posix::sep);
    }

    unsafe fn set_filename_unchecked(&mut self, filename: &[u8]) {
        match self.sepidx {
            None if bytes!("..") == self.repr => {
                let mut v = vec::with_capacity(3 + filename.len());
                v.push_all(dot_dot_static);
                v.push(posix::sep);
                v.push_all(filename);
                self.repr = PosixPath::normalize(v);
            }
            None => {
                self.repr = PosixPath::normalize(filename);
            }
            Some(idx) if self.repr.slice_from(idx+1) == bytes!("..") => {
                let mut v = vec::with_capacity(self.repr.len() + 1 + filename.len());
                v.push_all(self.repr);
                v.push(posix::sep);
                v.push_all(filename);
                self.repr = PosixPath::normalize(v);
            }
            Some(idx) => {
                let mut v = vec::with_capacity(self.repr.len() - idx + filename.len());
                v.push_all(self.repr.slice_to(idx+1));
                v.push_all(filename);
                self.repr = PosixPath::normalize(v);
            }
        }
        self.sepidx = self.repr.rposition_elem(&posix::sep);
    }

    unsafe fn push_unchecked(&mut self, path: &[u8]) {
        if !path.is_empty() {
            if path[0] == posix::sep {
                self.repr = PosixPath::normalize(path);
            }  else {
                let mut v = vec::with_capacity(self.repr.len() + path.len() + 1);
                v.push_all(self.repr);
                v.push(posix::sep);
                v.push_all(path);
                self.repr = PosixPath::normalize(v);
            }
            self.sepidx = self.repr.rposition_elem(&posix::sep);
        }
    }
}

impl GenericPath for PosixPath {
    #[inline]
    fn as_vec<'a>(&'a self) -> &'a [u8] {
        self.repr.as_slice()
    }

    fn dirname<'a>(&'a self) -> &'a [u8] {
        match self.sepidx {
            None if bytes!("..") == self.repr => self.repr.as_slice(),
            None => dot_static,
            Some(0) => self.repr.slice_to(1),
            Some(idx) if self.repr.slice_from(idx+1) == bytes!("..") => self.repr.as_slice(),
            Some(idx) => self.repr.slice_to(idx)
        }
    }

    fn filename<'a>(&'a self) -> &'a [u8] {
        match self.sepidx {
            None if bytes!(".") == self.repr || bytes!("..") == self.repr => &[],
            None => self.repr.as_slice(),
            Some(idx) if self.repr.slice_from(idx+1) == bytes!("..") => &[],
            Some(idx) => self.repr.slice_from(idx+1)
        }
    }

    fn pop_opt(&mut self) -> Option<~[u8]> {
        match self.sepidx {
            None if bytes!(".") == self.repr => None,
            None => {
                let mut v = ~['.' as u8];
                util::swap(&mut v, &mut self.repr);
                self.sepidx = None;
                Some(v)
            }
            Some(0) if bytes!("/") == self.repr => None,
            Some(idx) => {
                let v = self.repr.slice_from(idx+1).to_owned();
                if idx == 0 {
                    self.repr.truncate(idx+1);
                } else {
                    self.repr.truncate(idx);
                }
                self.sepidx = self.repr.rposition_elem(&posix::sep);
                Some(v)
            }
        }
    }

    #[inline]
    fn is_absolute(&self) -> bool {
        self.repr[0] == posix::sep
    }

    fn is_ancestor_of(&self, other: &PosixPath) -> bool {
        if self.is_absolute() != other.is_absolute() {
            false
        } else {
            let mut ita = self.component_iter();
            let mut itb = other.component_iter();
            if bytes!(".") == self.repr {
                return itb.next() != Some(bytes!(".."));
            }
            loop {
                match (ita.next(), itb.next()) {
                    (None, _) => break,
                    (Some(a), Some(b)) if a == b => { loop },
                    (Some(a), _) if a == bytes!("..") => {
                        // if ita contains only .. components, it's an ancestor
                        return ita.all(|x| x == bytes!(".."));
                    }
                    _ => return false
                }
            }
            true
        }
    }

    fn path_relative_from(&self, base: &PosixPath) -> Option<PosixPath> {
        if self.is_absolute() != base.is_absolute() {
            if self.is_absolute() {
                Some(self.clone())
            } else {
                None
            }
        } else {
            let mut ita = self.component_iter();
            let mut itb = base.component_iter();
            let mut comps = ~[];
            loop {
                match (ita.next(), itb.next()) {
                    (None, None) => break,
                    (Some(a), None) => {
                        comps.push(a);
                        comps.extend(&mut ita);
                        break;
                    }
                    (None, _) => comps.push(dot_dot_static),
                    (Some(a), Some(b)) if comps.is_empty() && a == b => (),
                    (Some(a), Some(b)) if b == bytes!(".") => comps.push(a),
                    (Some(_), Some(b)) if b == bytes!("..") => return None,
                    (Some(a), Some(_)) => {
                        comps.push(dot_dot_static);
                        for _ in itb {
                            comps.push(dot_dot_static);
                        }
                        comps.push(a);
                        comps.extend(&mut ita);
                        break;
                    }
                }
            }
            Some(PosixPath::new(comps.connect_vec(&posix::sep)))
        }
    }
}

impl PosixPath {
    /// Returns a new PosixPath from a byte vector
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the vector contains a NUL.
    #[inline]
    pub fn new(v: &[u8]) -> PosixPath {
        GenericPath::from_vec(v)
    }

    /// Returns a new PosixPath from a string
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the str contains a NUL.
    #[inline]
    pub fn from_str(s: &str) -> PosixPath {
        GenericPath::from_str(s)
    }

    /// Converts the PosixPath into an owned byte vector
    pub fn into_vec(self) -> ~[u8] {
        self.repr
    }

    /// Converts the PosixPath into an owned string, if possible
    pub fn into_str(self) -> Option<~str> {
        str::from_bytes_owned_opt(self.repr)
    }

    /// Returns a normalized byte vector representation of a path, by removing all empty
    /// components, and unnecessary . and .. components.
    pub fn normalize<V: Vector<u8>+CopyableVector<u8>>(v: V) -> ~[u8] {
        // borrowck is being very picky
        let val = {
            let is_abs = !v.as_slice().is_empty() && v.as_slice()[0] == posix::sep;
            let v_ = if is_abs { v.as_slice().slice_from(1) } else { v.as_slice() };
            let comps = normalize_helper(v_, is_abs, posix::is_sep);
            match comps {
                None => None,
                Some(comps) => {
                    if is_abs && comps.is_empty() {
                        Some(~[posix::sep])
                    } else {
                        let n = if is_abs { comps.len() } else { comps.len() - 1} +
                                comps.iter().map(|v| v.len()).sum();
                        let mut v = vec::with_capacity(n);
                        let mut it = comps.move_iter();
                        if !is_abs {
                            match it.next() {
                                None => (),
                                Some(comp) => v.push_all(comp)
                            }
                        }
                        for comp in it {
                            v.push(posix::sep);
                            v.push_all(comp);
                        }
                        Some(v)
                    }
                }
            }
        };
        match val {
            None => v.into_owned(),
            Some(val) => val
        }
    }

    /// Returns an iterator that yields each component of the path in turn.
    /// Does not distinguish between absolute and relative paths, e.g.
    /// /a/b/c and a/b/c yield the same set of components.
    /// A path of "/" yields no components. A path of "." yields one component.
    pub fn component_iter<'a>(&'a self) -> PosixComponentIter<'a> {
        let v = if self.repr[0] == posix::sep {
            self.repr.slice_from(1)
        } else { self.repr.as_slice() };
        let mut ret = v.split_iter(posix::is_sep);
        if v.is_empty() {
            // consume the empty "" component
            ret.next();
        }
        ret
    }
}

// None result means the byte vector didn't need normalizing
fn normalize_helper<'a>(v: &'a [u8], is_abs: bool, f: &'a fn(&u8) -> bool) -> Option<~[&'a [u8]]> {
    if is_abs && v.as_slice().is_empty() {
        return None;
    }
    let mut comps: ~[&'a [u8]] = ~[];
    let mut n_up = 0u;
    let mut changed = false;
    for comp in v.split_iter(f) {
        if comp.is_empty() { changed = true }
        else if comp == bytes!(".") { changed = true }
        else if comp == bytes!("..") {
            if is_abs && comps.is_empty() { changed = true }
            else if comps.len() == n_up { comps.push(dot_dot_static); n_up += 1 }
            else { comps.pop_opt(); changed = true }
        } else { comps.push(comp) }
    }
    if changed {
        if comps.is_empty() && !is_abs {
            if v == bytes!(".") {
                return None;
            }
            comps.push(dot_static);
        }
        Some(comps)
    } else {
        None
    }
}

static dot_static: &'static [u8] = &'static ['.' as u8];
static dot_dot_static: &'static [u8] = &'static ['.' as u8, '.' as u8];

/// Various POSIX helpers
pub mod posix {
    /// The standard path separator character
    pub static sep: u8 = '/' as u8;

    /// Returns whether the given byte is a path separator
    #[inline]
    pub fn is_sep(u: &u8) -> bool {
        *u == sep
    }
}

/// Various Windows helpers
pub mod windows {
    /// The standard path separator character
    pub static sep: u8 = '\\' as u8;
    /// The alternative path separator character
    pub static sep2: u8 = '/' as u8;

    /// Returns whether the given byte is a path separator
    #[inline]
    pub fn is_sep(u: &u8) -> bool {
        *u == sep || *u == sep2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use option::{Some, None};
    use iterator::Iterator;
    use vec::Vector;

    macro_rules! t(
        ($path:expr, $exp:expr) => (
            {
                let path = $path;
                assert_eq!(path.as_str(), $exp);
            }
        )
    )

    #[test]
    fn test_posix_paths() {
        t!(PosixPath::new(""), ".");
        t!(PosixPath::new("/"), "/");
        t!(PosixPath::new("hi"), "hi");
        t!(PosixPath::new("/lib"), "/lib");
        t!(PosixPath::new("hi/there"), "hi/there");
        t!(PosixPath::new("hi/there.txt"), "hi/there.txt");

        t!(PosixPath::new("hi/there/"), "hi/there");
        t!(PosixPath::new("hi/../there"), "there");
        t!(PosixPath::new("../hi/there"), "../hi/there");
        t!(PosixPath::new("/../hi/there"), "/hi/there");
        t!(PosixPath::new("foo/.."), ".");
        t!(PosixPath::new("/foo/.."), "/");
        t!(PosixPath::new("/foo/../.."), "/");
        t!(PosixPath::new("/foo/../../bar"), "/bar");
        t!(PosixPath::new("/./hi/./there/."), "/hi/there");
        t!(PosixPath::new("/./hi/./there/./.."), "/hi");
        t!(PosixPath::new("foo/../.."), "..");
        t!(PosixPath::new("foo/../../.."), "../..");
        t!(PosixPath::new("foo/../../bar"), "../bar");

        assert_eq!(PosixPath::new("foo/bar").into_str(), ~"foo/bar");
        assert_eq!(PosixPath::new("/foo/../../bar").into_str(), ~"/bar");
    }

    #[test]
    fn test_posix_components() {
        macro_rules! t(
            ($path:expr, $op:ident, $exp:expr) => (
                {
                    let path = PosixPath::new($path);
                    assert_eq!(path.$op(), $exp);
                }
            )
        )

        t!("a/b/c", filename, "c");
        t!("/a/b/c", filename, "c");
        t!("a", filename, "a");
        t!("/a", filename, "a");
        t!(".", filename, "");
        t!("/", filename, "");
        t!("..", filename, "");
        t!("../..", filename, "");

        t!("a/b/c", dirname, "a/b");
        t!("/a/b/c", dirname, "/a/b");
        t!("a", dirname, ".");
        t!("/a", dirname, "/");
        t!(".", dirname, ".");
        t!("/", dirname, "/");
        t!("..", dirname, "..");
        t!("../..", dirname, "../..");

        t!("hi/there.txt", filestem, "there");
        t!("hi/there", filestem, "there");
        t!("there.txt", filestem, "there");
        t!("there", filestem, "there");
        t!(".", filestem, "");
        t!("/", filestem, "");
        t!("foo/.bar", filestem, ".bar");
        t!(".bar", filestem, ".bar");
        t!("..bar", filestem, ".");
        t!("hi/there..txt", filestem, "there.");
        t!("..", filestem, "");
        t!("../..", filestem, "");

        t!("hi/there.txt", extension, Some("txt"));
        t!("hi/there", extension, None);
        t!("there.txt", extension, Some("txt"));
        t!("there", extension, None);
        t!(".", extension, None);
        t!("/", extension, None);
        t!("foo/.bar", extension, None);
        t!(".bar", extension, None);
        t!("..bar", extension, Some("bar"));
        t!("hi/there..txt", extension, Some("txt"));
        t!("..", extension, None);
        t!("../..", extension, None);
    }

    #[test]
    fn test_posix_push() {
        macro_rules! t(
            ($path:expr, $join:expr) => (
                {
                    let path = ($path);
                    let join = ($join);
                    let mut p1 = PosixPath::new(path);
                    p1.push(join);
                    let p2 = PosixPath::new(path);
                    assert_eq!(p1, p2.join(join));
                }
            )
        )

        t!("a/b/c", "..");
        t!("/a/b/c", "d");
        t!("a/b", "c/d");
        t!("a/b", "/c/d");
    }

    #[test]
    fn test_posix_push_path() {
        macro_rules! t(
            ($path:expr, $push:expr, $exp:expr) => (
                {
                    let mut p = PosixPath::new($path);
                    let push = PosixPath::new($push);
                    p.push_path(&push);
                    assert_eq!(p.as_str(), $exp);
                }
            )
        )

        t!("a/b/c", "d", "a/b/c/d");
        t!("/a/b/c", "d", "/a/b/c/d");
        t!("a/b", "c/d", "a/b/c/d");
        t!("a/b", "/c/d", "/c/d");
        t!("a/b", ".", "a/b");
        t!("a/b", "../c", "a/c");
    }

    #[test]
    fn test_posix_pop() {
        macro_rules! t(
            ($path:expr, $left:expr, $right:expr) => (
                {
                    let mut p = PosixPath::new($path);
                    let file = p.pop_opt();
                    assert_eq!(p.as_str(), $left);
                    assert_eq!(file.map(|s| s.as_slice()), $right);
                }
            )
        )

        t!("a/b/c", "a/b", Some("c"));
        t!("a", ".", Some("a"));
        t!(".", ".", None);
        t!("/a", "/", Some("a"));
        t!("/", "/", None);
    }

    #[test]
    fn test_posix_join() {
        t!(PosixPath::new("a/b/c").join(".."), "a/b");
        t!(PosixPath::new("/a/b/c").join("d"), "/a/b/c/d");
        t!(PosixPath::new("a/b").join("c/d"), "a/b/c/d");
        t!(PosixPath::new("a/b").join("/c/d"), "/c/d");
        t!(PosixPath::new(".").join("a/b"), "a/b");
        t!(PosixPath::new("/").join("a/b"), "/a/b");
    }

    #[test]
    fn test_posix_join_path() {
        macro_rules! t(
            ($path:expr, $join:expr, $exp:expr) => (
                {
                    let path = PosixPath::new($path);
                    let join = PosixPath::new($join);
                    let res = path.join_path(&join);
                    assert_eq!(res.as_str(), $exp);
                }
            )
        )

        t!("a/b/c", "..", "a/b");
        t!("/a/b/c", "d", "/a/b/c/d");
        t!("a/b", "c/d", "a/b/c/d");
        t!("a/b", "/c/d", "/c/d");
        t!(".", "a/b", "a/b");
        t!("/", "a/b", "/a/b");
    }

    #[test]
    fn test_posix_with_helpers() {
        t!(PosixPath::new("a/b/c").with_dirname("d"), "d/c");
        t!(PosixPath::new("a/b/c").with_dirname("d/e"), "d/e/c");
        t!(PosixPath::new("a/b/c").with_dirname(""), "c");
        t!(PosixPath::new("a/b/c").with_dirname("/"), "/c");
        t!(PosixPath::new("a/b/c").with_dirname("."), "c");
        t!(PosixPath::new("a/b/c").with_dirname(".."), "../c");
        t!(PosixPath::new("/").with_dirname("foo"), "foo");
        t!(PosixPath::new("/").with_dirname(""), ".");
        t!(PosixPath::new("/foo").with_dirname("bar"), "bar/foo");
        t!(PosixPath::new("..").with_dirname("foo"), "foo");
        t!(PosixPath::new("../..").with_dirname("foo"), "foo");
        t!(PosixPath::new("foo").with_dirname(".."), "../foo");
        t!(PosixPath::new("foo").with_dirname("../.."), "../../foo");

        t!(PosixPath::new("a/b/c").with_filename("d"), "a/b/d");
        t!(PosixPath::new(".").with_filename("foo"), "foo");
        t!(PosixPath::new("/a/b/c").with_filename("d"), "/a/b/d");
        t!(PosixPath::new("/").with_filename("foo"), "/foo");
        t!(PosixPath::new("/a").with_filename("foo"), "/foo");
        t!(PosixPath::new("foo").with_filename("bar"), "bar");
        t!(PosixPath::new("a/b/c").with_filename(""), "a/b");
        t!(PosixPath::new("a/b/c").with_filename("."), "a/b");
        t!(PosixPath::new("a/b/c").with_filename(".."), "a");
        t!(PosixPath::new("/a").with_filename(""), "/");
        t!(PosixPath::new("foo").with_filename(""), ".");
        t!(PosixPath::new("a/b/c").with_filename("d/e"), "a/b/d/e");
        t!(PosixPath::new("a/b/c").with_filename("/d"), "a/b/d");
        t!(PosixPath::new("..").with_filename("foo"), "../foo");
        t!(PosixPath::new("../..").with_filename("foo"), "../../foo");

        t!(PosixPath::new("hi/there.txt").with_filestem("here"), "hi/here.txt");
        t!(PosixPath::new("hi/there.txt").with_filestem(""), "hi/.txt");
        t!(PosixPath::new("hi/there.txt").with_filestem("."), "hi/..txt");
        t!(PosixPath::new("hi/there.txt").with_filestem(".."), "hi/...txt");
        t!(PosixPath::new("hi/there.txt").with_filestem("/"), "hi/.txt");
        t!(PosixPath::new("hi/there.txt").with_filestem("foo/bar"), "hi/foo/bar.txt");
        t!(PosixPath::new("hi/there.foo.txt").with_filestem("here"), "hi/here.txt");
        t!(PosixPath::new("hi/there").with_filestem("here"), "hi/here");
        t!(PosixPath::new("hi/there").with_filestem(""), "hi");
        t!(PosixPath::new("hi").with_filestem(""), ".");
        t!(PosixPath::new("/hi").with_filestem(""), "/");
        t!(PosixPath::new("hi/there").with_filestem(".."), ".");
        t!(PosixPath::new("hi/there").with_filestem("."), "hi");
        t!(PosixPath::new("hi/there.").with_filestem("foo"), "hi/foo.");
        t!(PosixPath::new("hi/there.").with_filestem(""), "hi");
        t!(PosixPath::new("hi/there.").with_filestem("."), ".");
        t!(PosixPath::new("hi/there.").with_filestem(".."), "hi/...");
        t!(PosixPath::new("/").with_filestem("foo"), "/foo");
        t!(PosixPath::new(".").with_filestem("foo"), "foo");
        t!(PosixPath::new("hi/there..").with_filestem("here"), "hi/here.");
        t!(PosixPath::new("hi/there..").with_filestem(""), "hi");

        t!(PosixPath::new("hi/there.txt").with_extension("exe"), "hi/there.exe");
        t!(PosixPath::new("hi/there.txt").with_extension(""), "hi/there");
        t!(PosixPath::new("hi/there.txt").with_extension("."), "hi/there..");
        t!(PosixPath::new("hi/there.txt").with_extension(".."), "hi/there...");
        t!(PosixPath::new("hi/there").with_extension("txt"), "hi/there.txt");
        t!(PosixPath::new("hi/there").with_extension("."), "hi/there..");
        t!(PosixPath::new("hi/there").with_extension(".."), "hi/there...");
        t!(PosixPath::new("hi/there.").with_extension("txt"), "hi/there.txt");
        t!(PosixPath::new("hi/.foo").with_extension("txt"), "hi/.foo.txt");
        t!(PosixPath::new("hi/there.txt").with_extension(".foo"), "hi/there..foo");
        t!(PosixPath::new("/").with_extension("txt"), "/");
        t!(PosixPath::new("/").with_extension("."), "/");
        t!(PosixPath::new("/").with_extension(".."), "/");
        t!(PosixPath::new(".").with_extension("txt"), ".");
    }

    #[test]
    fn test_posix_setters() {
        macro_rules! t(
            ($path:expr, $set:ident, $with:ident, $arg:expr) => (
                {
                    let path = ($path);
                    let arg = ($arg);
                    let mut p1 = PosixPath::new(path);
                    p1.$set(arg);
                    let p2 = PosixPath::new(path);
                    assert_eq!(p1, p2.$with(arg));
                }
            )
        )

        t!("a/b/c", set_dirname, with_dirname, "d");
        t!("a/b/c", set_dirname, with_dirname, "d/e");
        t!("/", set_dirname, with_dirname, "foo");
        t!("/foo", set_dirname, with_dirname, "bar");
        t!("a/b/c", set_dirname, with_dirname, "");
        t!("../..", set_dirname, with_dirname, "x");
        t!("foo", set_dirname, with_dirname, "../..");

        t!("a/b/c", set_filename, with_filename, "d");
        t!("/", set_filename, with_filename, "foo");
        t!(".", set_filename, with_filename, "foo");
        t!("a/b", set_filename, with_filename, "");
        t!("a", set_filename, with_filename, "");

        t!("hi/there.txt", set_filestem, with_filestem, "here");
        t!("hi/there.", set_filestem, with_filestem, "here");
        t!("hi/there", set_filestem, with_filestem, "here");
        t!("hi/there.txt", set_filestem, with_filestem, "");
        t!("hi/there", set_filestem, with_filestem, "");

        t!("hi/there.txt", set_extension, with_extension, "exe");
        t!("hi/there.", set_extension, with_extension, "txt");
        t!("hi/there", set_extension, with_extension, "txt");
        t!("hi/there.txt", set_extension, with_extension, "");
        t!("hi/there", set_extension, with_extension, "");
        t!(".", set_extension, with_extension, "txt");
    }

    #[test]
    fn test_posix_dir_file_path() {
        t!(PosixPath::new("hi/there").dir_path(), "hi");
        t!(PosixPath::new("hi").dir_path(), ".");
        t!(PosixPath::new("/hi").dir_path(), "/");
        t!(PosixPath::new("/").dir_path(), "/");
        t!(PosixPath::new("..").dir_path(), "..");
        t!(PosixPath::new("../..").dir_path(), "../..");

        macro_rules! t(
            ($path:expr, $exp:expr) => (
                {
                    let path = $path;
                    let left = path.map(|p| p.as_str());
                    assert_eq!(left, $exp);
                }
            )
        )

        t!(PosixPath::new("hi/there").file_path(), Some("there"));
        t!(PosixPath::new("hi").file_path(), Some("hi"));
        t!(PosixPath::new(".").file_path(), None);
        t!(PosixPath::new("/").file_path(), None);
        t!(PosixPath::new("..").file_path(), None);
        t!(PosixPath::new("../..").file_path(), None);
    }

    #[test]
    fn test_posix_is_absolute() {
        assert_eq!(PosixPath::new("a/b/c").is_absolute(), false);
        assert_eq!(PosixPath::new("/a/b/c").is_absolute(), true);
        assert_eq!(PosixPath::new("a").is_absolute(), false);
        assert_eq!(PosixPath::new("/a").is_absolute(), true);
        assert_eq!(PosixPath::new(".").is_absolute(), false);
        assert_eq!(PosixPath::new("/").is_absolute(), true);
        assert_eq!(PosixPath::new("..").is_absolute(), false);
        assert_eq!(PosixPath::new("../..").is_absolute(), false);
    }

    #[test]
    fn test_posix_is_ancestor_of() {
        macro_rules! t(
            ($path:expr, $dest:expr, $exp:expr) => (
                {
                    let path = PosixPath::new($path);
                    let dest = PosixPath::new($dest);
                    assert_eq!(path.is_ancestor_of(&dest), $exp);
                }
            )
        )

        t!("a/b/c", "a/b/c/d", true);
        t!("a/b/c", "a/b/c", true);
        t!("a/b/c", "a/b", false);
        t!("/a/b/c", "/a/b/c", true);
        t!("/a/b", "/a/b/c", true);
        t!("/a/b/c/d", "/a/b/c", false);
        t!("/a/b", "a/b/c", false);
        t!("a/b", "/a/b/c", false);
        t!("a/b/c", "a/b/d", false);
        t!("../a/b/c", "a/b/c", false);
        t!("a/b/c", "../a/b/c", false);
        t!("a/b/c", "a/b/cd", false);
        t!("a/b/cd", "a/b/c", false);
        t!("../a/b", "../a/b/c", true);
        t!(".", "a/b", true);
        t!(".", ".", true);
        t!("/", "/", true);
        t!("/", "/a/b", true);
        t!("..", "a/b", true);
        t!("../..", "a/b", true);
    }

    #[test]
    fn test_posix_path_relative_from() {
        macro_rules! t(
            ($path:expr, $other:expr, $exp:expr) => (
                {
                    let path = PosixPath::new($path);
                    let other = PosixPath::new($other);
                    let res = path.path_relative_from(&other);
                    assert_eq!(res.map(|x| x.as_str()), $exp);
                }
            )
        )

        t!("a/b/c", "a/b", Some("c"));
        t!("a/b/c", "a/b/d", Some("../c"));
        t!("a/b/c", "a/b/c/d", Some(".."));
        t!("a/b/c", "a/b/c", Some("."));
        t!("a/b/c", "a/b/c/d/e", Some("../.."));
        t!("a/b/c", "a/d/e", Some("../../b/c"));
        t!("a/b/c", "d/e/f", Some("../../../a/b/c"));
        t!("a/b/c", "/a/b/c", None);
        t!("/a/b/c", "a/b/c", Some("/a/b/c"));
        t!("/a/b/c", "/a/b/c/d", Some(".."));
        t!("/a/b/c", "/a/b", Some("c"));
        t!("/a/b/c", "/a/b/c/d/e", Some("../.."));
        t!("/a/b/c", "/a/d/e", Some("../../b/c"));
        t!("/a/b/c", "/d/e/f", Some("../../../a/b/c"));
        t!("hi/there.txt", "hi/there", Some("../there.txt"));
        t!(".", "a", Some(".."));
        t!(".", "a/b", Some("../.."));
        t!(".", ".", Some("."));
        t!("a", ".", Some("a"));
        t!("a/b", ".", Some("a/b"));
        t!("..", ".", Some(".."));
        t!("a/b/c", "a/b/c", Some("."));
        t!("/a/b/c", "/a/b/c", Some("."));
        t!("/", "/", Some("."));
        t!("/", ".", Some("/"));
        t!("../../a", "b", Some("../../../a"));
        t!("a", "../../b", None);
        t!("../../a", "../../b", Some("../a"));
        t!("../../a", "../../a/b", Some(".."));
        t!("../../a/b", "../../a", Some("b"));
    }

    #[test]
    fn test_posix_component_iter() {
        macro_rules! t(
            ($path:expr, $exp:expr) => (
                {
                    let path = PosixPath::new($path);
                    let comps = path.component_iter().to_owned_vec();
                    assert_eq!(comps.as_slice(), $exp);
                }
            )
        )

        t!("a/b/c", ["a", "b", "c"]);
        t!("a/b/d", ["a", "b", "d"]);
        t!("a/b/cd", ["a", "b", "cd"]);
        t!("/a/b/c", ["a", "b", "c"]);
        t!("a", ["a"]);
        t!("/a", ["a"]);
        t!("/", []);
        t!(".", ["."]);
        t!("..", [".."]);
        t!("../..", ["..", ".."]);
        t!("../../foo", ["..", "..", "foo"]);
    }
}
