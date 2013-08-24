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

use c_str::{CString, ToCStr};
use clone::Clone;
use cmp::Eq;
use from_str::FromStr;
use option::{Option, None, Some};
use str;
use str::{OwnedStr, Str, StrSlice};
use to_str::ToStr;

/// Typedef for the platform-native path type
#[cfg(unix)]
pub type Path = PosixPath;
// /// Typedef for the platform-native path type
//#[cfg(windows)]
//pub type Path = WindowsPath;

/// Represents a POSIX file path
#[deriving(Clone, DeepClone)]
pub struct PosixPath {
    priv repr: ~str, // assumed to never be empty
    priv sepidx: Option<uint> // index of the final separator in repr
}

impl Eq for PosixPath {
    fn eq(&self, other: &PosixPath) -> bool {
        self.repr == other.repr
    }
}

impl FromStr for PosixPath {
    fn from_str(s: &str) -> Option<PosixPath> {
        Some(PosixPath::new(s))
    }
}

/// A trait that represents the generic operations available on paths
pub trait GenericPath: Clone {
    /// Creates a new Path from a string.
    /// The resulting path will always be normalized.
    fn from_str(path: &str) -> Self;

    /// Returns the path as a string
    fn as_str<'a>(&'a self) -> &'a str;

    /// Returns the directory component of `self`, as a string (with no trailing separator).
    /// If `self` has no directory component, returns ".".
    fn dirname<'a>(&'a self) -> &'a str;
    /// Returns the file component of `self`, as a string.
    /// If `self` represents the root of the file hierarchy, returns the empty string.
    /// If `self` is ".", returns the empty string.
    fn filename<'a>(&'a self) -> &'a str;
    /// Returns the stem of the filename of `self`, as a string.
    /// The stem is the portion of the filename just before the last '.'.
    /// If there is no '.', the entire filename is returned.
    fn filestem<'a>(&'a self) -> &'a str {
        let name = self.filename();
        match name.rfind('.') {
            None | Some(0) => name,
            Some(1) if name == ".." => name,
            Some(pos) => name.slice_to(pos)
        }
    }
    /// Returns the extension of the filename of `self`, as a string option.
    /// The extension is the portion of the filename just after the last '.'.
    /// If there is no extension, None is returned.
    /// If the filename ends in '.', the empty string is returned.
    fn extension<'a>(&'a self) -> Option<&'a str> {
        let name = self.filename();
        match name.rfind('.') {
            None | Some(0) => None,
            Some(1) if name == ".." => None,
            Some(pos) => Some(name.slice_from(pos+1))
        }
    }

    /// Replaces the directory portion of the path with the given string.
    /// If `self` represents the root of the filesystem hierarchy, the last path component
    /// of the given string becomes the filename.
    fn set_dirname(&mut self, dirname: &str);
    /// Replaces the filename portion of the path with the given string.
    /// If the replacement name is "", this is equivalent to popping the path.
    fn set_filename(&mut self, filename: &str);
    /// Replaces the filestem with the given string.
    /// If there is no extension in `self` (or `self` has no filename), this is equivalent
    /// to `set_filename`. Otherwise, if the given string is "", the extension (including
    /// the preceding ".") becomes the new filename.
    fn set_filestem(&mut self, filestem: &str) {
        // borrowck is being a pain here
        let val = {
            let name = self.filename();
            if !name.is_empty() {
                match name.rfind('.') {
                    None | Some(0) => None,
                    Some(idx) => {
                        let mut s = str::with_capacity(filestem.len() + name.len() - idx);
                        s.push_str(filestem);
                        s.push_str(name.slice_from(idx));
                        Some(s)
                    }
                }
            } else { None }
        };
        match val {
            None => self.set_filename(filestem),
            Some(s) => self.set_filename(s)
        }
    }
    /// Replaces the extension with the given string.
    /// If there is no extension in `self`, this adds one.
    /// If the given string is "", this removes the extension.
    /// If `self` has no filename, this is a no-op.
    fn set_extension(&mut self, extension: &str) {
        // borrowck causes problems here too
        let val = {
            let name = self.filename();
            if !name.is_empty() {
                match name.rfind('.') {
                    None | Some(0) => {
                        if extension.is_empty() {
                            None
                        } else {
                            let mut s = str::with_capacity(name.len() + extension.len() + 1);
                            s.push_str(name);
                            s.push_char('.');
                            s.push_str(extension);
                            Some(s)
                        }
                    }
                    Some(idx) => {
                        if extension.is_empty() {
                            Some(name.slice_to(idx).to_owned())
                        } else {
                            let mut s = str::with_capacity(idx + extension.len() + 1);
                            s.push_str(name.slice_to(idx+1));
                            s.push_str(extension);
                            Some(s)
                        }
                    }
                }
            } else { None }
        };
        match val {
            None => (),
            Some(s) => self.set_filename(s)
        }
    }

    /// Returns a new Path constructed by replacing the dirname with the given string.
    /// See `set_dirname` for details.
    fn with_dirname(&self, dirname: &str) -> Self {
        let mut p = self.clone();
        p.set_dirname(dirname);
        p
    }
    /// Returns a new Path constructed by replacing the filename with the given string.
    /// See `set_filename` for details.
    fn with_filename(&self, filename: &str) -> Self {
        let mut p = self.clone();
        p.set_filename(filename);
        p
    }
    /// Returns a new Path constructed by setting the filestem to the given string.
    /// See `set_filestem` for details.
    fn with_filestem(&self, filestem: &str) -> Self {
        let mut p = self.clone();
        p.set_filestem(filestem);
        p
    }
    /// Returns a new Path constructed by setting the extension to the given string.
    /// See `set_extension` for details.
    fn with_extension(&self, extension: &str) -> Self {
        let mut p = self.clone();
        p.set_extension(extension);
        p
    }


    /// Returns the directory component of `self`, as a Path.
    /// If `self` represents the root of the filesystem hierarchy, returns `self`.
    fn dir_path(&self) -> Self {
        GenericPath::from_str(self.dirname())
    }
    /// Returns the file component of `self`, as a relative Path.
    /// If `self` represents the root of the filesystem hierarchy, returns None.
    fn file_path(&self) -> Option<Self> {
        match self.filename() {
            "" => None,
            s => Some(GenericPath::from_str(s))
        }
    }

    /// Pushes a path (as a string) onto `self`.
    /// If the argument represents an absolute path, it replaces `self`.
    fn push(&mut self, path: &str);
    /// Pushes a Path onto `self`.
    /// If the argument represents an absolute path, it replaces `self`.
    fn push_path(&mut self, path: &Self);
    /// Pops the last path component off of `self` and returns it.
    /// If `self` represents the root of the file hierarchy, None is returned.
    fn pop_opt(&mut self) -> Option<~str>;

    /// Returns a new Path constructed by joining `self` with the given path (as a string).
    /// If the given path is absolute, the new Path will represent just that.
    fn join(&self, path: &str) -> Self {
        let mut p = self.clone();
        p.push(path);
        p
    }
    /// Returns a new Path constructed by joining `self` with the given path.
    /// If the given path is absolute, the new Path will represent just that.
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

impl ToStr for PosixPath {
    #[inline]
    fn to_str(&self) -> ~str {
        self.as_str().to_owned()
    }
}

impl ToCStr for PosixPath {
    #[inline]
    fn to_c_str(&self) -> CString {
        self.as_str().to_c_str()
    }

    #[inline]
    unsafe fn to_c_str_unchecked(&self) -> CString {
        self.as_str().to_c_str_unchecked()
    }
}

/// Various POSIX helpers
pub mod posix {
    /// The standard path separator character
    pub static sep: char = '/';

    /// Returns whether the given char is a path separator
    #[inline]
    pub fn is_sep(u: char) -> bool {
        u == sep
    }
}

/// Various Windows helpers
pub mod windows {
    /// The standard path separator character
    pub static sep: char = '\\';

    /// Returns whether the given char is a path separator (both / and \)
    #[inline]
    pub fn is_sep(u: char) -> bool {
        u == sep || u == '/'
    }
}
