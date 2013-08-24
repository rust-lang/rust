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
use str::{OwnedStr, Str, StrSlice, StrVector};
use to_str::ToStr;
use util;
use vec::{ImmutableVector, OwnedVector};

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
type PosixComponentIter<'self> = str::CharSplitIterator<'self, char>;

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

impl GenericPath for PosixPath {
    #[inline]
    fn from_str(s: &str) -> PosixPath {
        PosixPath::new(s)
    }

    #[inline]
    fn as_str<'a>(&'a self) -> &'a str {
        self.repr.as_slice()
    }

    fn dirname<'a>(&'a self) -> &'a str {
        match self.sepidx {
            None if ".." == self.repr => "..",
            None => ".",
            Some(0) => self.repr.slice_to(1),
            Some(idx) if self.repr.slice_from(idx+1) == ".." => self.repr.as_slice(),
            Some(idx) => self.repr.slice_to(idx)
        }
    }

    fn filename<'a>(&'a self) -> &'a str {
        match self.sepidx {
            None if "." == self.repr || ".." == self.repr => "",
            None => self.repr.as_slice(),
            Some(idx) if self.repr.slice_from(idx+1) == ".." => "",
            Some(idx) => self.repr.slice_from(idx+1)
        }
    }

    fn set_dirname(&mut self, dirname: &str) {
        match self.sepidx {
            None if "." == self.repr || ".." == self.repr => {
                self.repr = PosixPath::normalize(dirname);
            }
            None => {
                let mut s = str::with_capacity(dirname.len() + self.repr.len() + 1);
                s.push_str(dirname);
                s.push_char(posix::sep);
                s.push_str(self.repr);
                self.repr = PosixPath::normalize(s);
            }
            Some(0) if self.repr.len() == 1 && self.repr[0] == posix::sep as u8 => {
                self.repr = PosixPath::normalize(dirname);
            }
            Some(idx) if dirname == "" => {
                let s = PosixPath::normalize(self.repr.slice_from(idx+1));
                self.repr = s;
            }
            Some(idx) if self.repr.slice_from(idx+1) == ".." => {
                self.repr = PosixPath::normalize(dirname);
            }
            Some(idx) => {
                let mut s = str::with_capacity(dirname.len() + self.repr.len() - idx);
                s.push_str(dirname);
                s.push_str(self.repr.slice_from(idx));
                self.repr = PosixPath::normalize(s);
            }
        }
        self.sepidx = self.repr.rfind(posix::sep);
    }

    fn set_filename(&mut self, filename: &str) {
        match self.sepidx {
            None if ".." == self.repr => {
                let mut s = str::with_capacity(3 + filename.len());
                s.push_str("..");
                s.push_char(posix::sep);
                s.push_str(filename);
                self.repr = PosixPath::normalize(s);
            }
            None => {
                self.repr = PosixPath::normalize(filename);
            }
            Some(idx) if self.repr.slice_from(idx+1) == ".." => {
                let mut s = str::with_capacity(self.repr.len() + 1 + filename.len());
                s.push_str(self.repr);
                s.push_char(posix::sep);
                s.push_str(filename);
                self.repr = PosixPath::normalize(s);
            }
            Some(idx) => {
                let mut s = str::with_capacity(self.repr.len() - idx + filename.len());
                s.push_str(self.repr.slice_to(idx+1));
                s.push_str(filename);
                self.repr = PosixPath::normalize(s);
            }
        }
        self.sepidx = self.repr.rfind(posix::sep);
    }

    fn push(&mut self, path: &str) {
        if !path.is_empty() {
            if path[0] == posix::sep as u8 {
                self.repr = PosixPath::normalize(path);
            }  else {
                let mut s = str::with_capacity(self.repr.len() + path.len() + 1);
                s.push_str(self.repr);
                s.push_char(posix::sep);
                s.push_str(path);
                self.repr = PosixPath::normalize(s);
            }
            self.sepidx = self.repr.rfind(posix::sep);
        }
    }

    fn push_path(&mut self, path: &PosixPath) {
        self.push(path.as_str());
    }

    fn pop_opt(&mut self) -> Option<~str> {
        match self.sepidx {
            None if "." == self.repr => None,
            None => {
                let mut s = ~".";
                util::swap(&mut s, &mut self.repr);
                self.sepidx = None;
                Some(s)
            }
            Some(0) if "/" == self.repr => None,
            Some(idx) => {
                let s = self.repr.slice_from(idx+1).to_owned();
                if idx == 0 {
                    self.repr.truncate(idx+1);
                } else {
                    self.repr.truncate(idx);
                }
                self.sepidx = self.repr.rfind(posix::sep);
                Some(s)
            }
        }
    }

    #[inline]
    fn is_absolute(&self) -> bool {
        self.repr[0] == posix::sep as u8
    }

    fn is_ancestor_of(&self, other: &PosixPath) -> bool {
        if self.is_absolute() != other.is_absolute() {
            false
        } else {
            let mut ita = self.component_iter();
            let mut itb = other.component_iter();
            if "." == self.repr {
                return match itb.next() {
                    Some("..") => false,
                    _ => true
                };
            }
            loop {
                match (ita.next(), itb.next()) {
                    (None, _) => break,
                    (Some(a), Some(b)) if a == b => { loop },
                    (Some(".."), _) => {
                        // if ita contains only .. components, it's an ancestor
                        return ita.all(|x| x == "..");
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
                    (None, _) => comps.push(".."),
                    (Some(a), Some(b)) if comps.is_empty() && a == b => (),
                    (Some(a), Some(".")) => comps.push(a),
                    (Some(_), Some("..")) => return None,
                    (Some(a), Some(_)) => {
                        comps.push("..");
                        for _ in itb {
                            comps.push("..");
                        }
                        comps.push(a);
                        comps.extend(&mut ita);
                        break;
                    }
                }
            }
            Some(PosixPath::new(comps.connect(str::from_char(posix::sep))))
        }
    }
}

impl PosixPath {
    /// Returns a new PosixPath from a string
    pub fn new(s: &str) -> PosixPath {
        let s = PosixPath::normalize(s);
        assert!(!s.is_empty());
        let idx = s.rfind(posix::sep);
        PosixPath{ repr: s, sepidx: idx }
    }

    /// Converts the PosixPath into an owned string
    pub fn into_str(self) -> ~str {
        self.repr
    }

    /// Returns a normalized string representation of a path, by removing all empty
    /// components, and unnecessary . and .. components.
    pub fn normalize<S: Str>(s: S) -> ~str {
        // borrowck is being very picky
        let val = {
            let is_abs = !s.as_slice().is_empty() && s.as_slice()[0] == posix::sep as u8;
            let s_ = if is_abs { s.as_slice().slice_from(1) } else { s.as_slice() };
            let comps = normalize_helper(s_, is_abs, posix::sep);
            match comps {
                None => None,
                Some(comps) => {
                    let sepstr = str::from_char(posix::sep);
                    if is_abs && comps.is_empty() {
                        Some(sepstr)
                    } else {
                        let n = if is_abs { comps.len() } else { comps.len() - 1} +
                                comps.iter().map(|s| s.len()).sum();
                        let mut s = str::with_capacity(n);
                        let mut it = comps.move_iter();
                        if !is_abs {
                            match it.next() {
                                None => (),
                                Some(comp) => s.push_str(comp)
                            }
                        }
                        for comp in it {
                            s.push_str(sepstr);
                            s.push_str(comp);
                        }
                        Some(s)
                    }
                }
            }
        };
        match val {
            None => s.into_owned(),
            Some(val) => val
        }
    }

    /// Returns an iterator that yields each component of the path in turn.
    /// Does not distinguish between absolute and relative paths, e.g.
    /// /a/b/c and a/b/c yield the same set of components.
    /// A path of "/" yields no components. A path of "." yields one component.
    pub fn component_iter<'a>(&'a self) -> PosixComponentIter<'a> {
        let s = if self.repr[0] == posix::sep as u8 {
            self.repr.slice_from(1)
        } else { self.repr.as_slice() };
        let mut ret = s.split_iter(posix::sep);
        if s.is_empty() {
            // consume the empty "" component
            ret.next();
        }
        ret
    }
}

// None result means the string didn't need normalizing
fn normalize_helper<'a, Sep: str::CharEq>(s: &'a str, is_abs: bool, sep: Sep) -> Option<~[&'a str]> {
    if is_abs && s.as_slice().is_empty() {
        return None;
    }
    let mut comps: ~[&'a str] = ~[];
    let mut n_up = 0u;
    let mut changed = false;
    for comp in s.split_iter(sep) {
        match comp {
            "" => { changed = true; }
            "." => { changed = true; }
            ".." if is_abs && comps.is_empty() => { changed = true; }
            ".." if comps.len() == n_up => { comps.push(".."); n_up += 1; }
            ".." => { comps.pop_opt(); changed = true; }
            x => comps.push(x)
        }
    }
    if changed {
        if comps.is_empty() && !is_abs {
            if s == "." {
                return None;
            }
            comps.push(".");
        }
        Some(comps)
    } else {
        None
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
