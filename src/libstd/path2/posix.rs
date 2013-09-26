// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! POSIX file path handling

use container::Container;
use c_str::{CString, ToCStr};
use clone::Clone;
use cmp::Eq;
use from_str::FromStr;
use iter::{AdditiveIterator, Extendable, Iterator};
use option::{Option, None, Some};
use str;
use str::Str;
use util;
use vec;
use vec::CopyableVector;
use vec::{Vector, VectorVector};
use super::{GenericPath, GenericPathUnsafe};

/// Iterator that yields successive components of a Path
pub type ComponentIter<'self> = vec::SplitIterator<'self, u8>;

/// Represents a POSIX file path
#[deriving(Clone, DeepClone)]
pub struct Path {
    priv repr: ~[u8], // assumed to never be empty or contain NULs
    priv sepidx: Option<uint> // index of the final separator in repr
}

/// The standard path separator character
pub static sep: u8 = '/' as u8;

/// Returns whether the given byte is a path separator
#[inline]
pub fn is_sep(u: &u8) -> bool {
    *u == sep
}

impl Eq for Path {
    #[inline]
    fn eq(&self, other: &Path) -> bool {
        self.repr == other.repr
    }
}

impl FromStr for Path {
    fn from_str(s: &str) -> Option<Path> {
        let v = s.as_bytes();
        if contains_nul(v) {
            None
        } else {
            Some(unsafe { GenericPathUnsafe::from_vec_unchecked(v) })
        }
    }
}

impl ToCStr for Path {
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

impl GenericPathUnsafe for Path {
    unsafe fn from_vec_unchecked(path: &[u8]) -> Path {
        let path = Path::normalize(path);
        assert!(!path.is_empty());
        let idx = path.rposition_elem(&sep);
        Path{ repr: path, sepidx: idx }
    }

    unsafe fn set_dirname_unchecked(&mut self, dirname: &[u8]) {
        match self.sepidx {
            None if bytes!(".") == self.repr || bytes!("..") == self.repr => {
                self.repr = Path::normalize(dirname);
            }
            None => {
                let mut v = vec::with_capacity(dirname.len() + self.repr.len() + 1);
                v.push_all(dirname);
                v.push(sep);
                v.push_all(self.repr);
                self.repr = Path::normalize(v);
            }
            Some(0) if self.repr.len() == 1 && self.repr[0] == sep => {
                self.repr = Path::normalize(dirname);
            }
            Some(idx) if self.repr.slice_from(idx+1) == bytes!("..") => {
                self.repr = Path::normalize(dirname);
            }
            Some(idx) if dirname.is_empty() => {
                let v = Path::normalize(self.repr.slice_from(idx+1));
                self.repr = v;
            }
            Some(idx) => {
                let mut v = vec::with_capacity(dirname.len() + self.repr.len() - idx);
                v.push_all(dirname);
                v.push_all(self.repr.slice_from(idx));
                self.repr = Path::normalize(v);
            }
        }
        self.sepidx = self.repr.rposition_elem(&sep);
    }

    unsafe fn set_filename_unchecked(&mut self, filename: &[u8]) {
        match self.sepidx {
            None if bytes!("..") == self.repr => {
                let mut v = vec::with_capacity(3 + filename.len());
                v.push_all(dot_dot_static);
                v.push(sep);
                v.push_all(filename);
                self.repr = Path::normalize(v);
            }
            None => {
                self.repr = Path::normalize(filename);
            }
            Some(idx) if self.repr.slice_from(idx+1) == bytes!("..") => {
                let mut v = vec::with_capacity(self.repr.len() + 1 + filename.len());
                v.push_all(self.repr);
                v.push(sep);
                v.push_all(filename);
                self.repr = Path::normalize(v);
            }
            Some(idx) => {
                let mut v = vec::with_capacity(idx + 1 + filename.len());
                v.push_all(self.repr.slice_to(idx+1));
                v.push_all(filename);
                self.repr = Path::normalize(v);
            }
        }
        self.sepidx = self.repr.rposition_elem(&sep);
    }

    unsafe fn push_unchecked(&mut self, path: &[u8]) {
        if !path.is_empty() {
            if path[0] == sep {
                self.repr = Path::normalize(path);
            }  else {
                let mut v = vec::with_capacity(self.repr.len() + path.len() + 1);
                v.push_all(self.repr);
                v.push(sep);
                v.push_all(path);
                self.repr = Path::normalize(v);
            }
            self.sepidx = self.repr.rposition_elem(&sep);
        }
    }
}

impl GenericPath for Path {
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
                self.sepidx = self.repr.rposition_elem(&sep);
                Some(v)
            }
        }
    }

    #[inline]
    fn is_absolute(&self) -> bool {
        self.repr[0] == sep
    }

    fn is_ancestor_of(&self, other: &Path) -> bool {
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

    fn path_relative_from(&self, base: &Path) -> Option<Path> {
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
            Some(Path::from_vec(comps.connect_vec(&sep)))
        }
    }
}

impl Path {
    /// Returns a new Path from a byte vector
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the vector contains a NUL.
    #[inline]
    pub fn from_vec(v: &[u8]) -> Path {
        GenericPath::from_vec(v)
    }

    /// Returns a new Path from a byte vector, if possible
    #[inline]
    pub fn from_vec_opt(v: &[u8]) -> Option<Path> {
        GenericPath::from_vec_opt(v)
    }

    /// Returns a new Path from a string
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the str contains a NUL.
    #[inline]
    pub fn from_str(s: &str) -> Path {
        GenericPath::from_str(s)
    }

    /// Returns a new Path from a string, if possible
    #[inline]
    pub fn from_str_opt(s: &str) -> Option<Path> {
        GenericPath::from_str_opt(s)
    }

    /// Converts the Path into an owned byte vector
    pub fn into_vec(self) -> ~[u8] {
        self.repr
    }

    /// Converts the Path into an owned string, if possible
    pub fn into_str(self) -> Option<~str> {
        str::from_utf8_owned_opt(self.repr)
    }

    /// Returns a normalized byte vector representation of a path, by removing all empty
    /// components, and unnecessary . and .. components.
    pub fn normalize<V: Vector<u8>+CopyableVector<u8>>(v: V) -> ~[u8] {
        // borrowck is being very picky
        let val = {
            let is_abs = !v.as_slice().is_empty() && v.as_slice()[0] == sep;
            let v_ = if is_abs { v.as_slice().slice_from(1) } else { v.as_slice() };
            let comps = normalize_helper(v_, is_abs);
            match comps {
                None => None,
                Some(comps) => {
                    if is_abs && comps.is_empty() {
                        Some(~[sep])
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
                            v.push(sep);
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
    pub fn component_iter<'a>(&'a self) -> ComponentIter<'a> {
        let v = if self.repr[0] == sep {
            self.repr.slice_from(1)
        } else { self.repr.as_slice() };
        let mut ret = v.split_iter(is_sep);
        if v.is_empty() {
            // consume the empty "" component
            ret.next();
        }
        ret
    }
}

// None result means the byte vector didn't need normalizing
fn normalize_helper<'a>(v: &'a [u8], is_abs: bool) -> Option<~[&'a [u8]]> {
    if is_abs && v.as_slice().is_empty() {
        return None;
    }
    let mut comps: ~[&'a [u8]] = ~[];
    let mut n_up = 0u;
    let mut changed = false;
    for comp in v.split_iter(is_sep) {
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

// FIXME (#8169): Pull this into parent module once visibility works
#[inline(always)]
fn contains_nul(v: &[u8]) -> bool {
    v.iter().any(|&x| x == 0)
}

static dot_static: &'static [u8] = &'static ['.' as u8];
static dot_dot_static: &'static [u8] = &'static ['.' as u8, '.' as u8];

#[cfg(test)]
mod tests {
    use super::*;
    use option::{Some, None};
    use iter::Iterator;
    use str;
    use vec::Vector;

    macro_rules! t(
        (s: $path:expr, $exp:expr) => (
            {
                let path = $path;
                assert_eq!(path.as_str(), Some($exp));
            }
        );
        (v: $path:expr, $exp:expr) => (
            {
                let path = $path;
                assert_eq!(path.as_vec(), $exp);
            }
        )
    )

    macro_rules! b(
        ($($arg:expr),+) => (
            bytes!($($arg),+)
        )
    )

    #[test]
    fn test_paths() {
        t!(v: Path::from_vec([]), b!("."));
        t!(v: Path::from_vec(b!("/")), b!("/"));
        t!(v: Path::from_vec(b!("a/b/c")), b!("a/b/c"));
        t!(v: Path::from_vec(b!("a/b/c", 0xff)), b!("a/b/c", 0xff));
        t!(v: Path::from_vec(b!(0xff, "/../foo", 0x80)), b!("foo", 0x80));
        let p = Path::from_vec(b!("a/b/c", 0xff));
        assert_eq!(p.as_str(), None);

        t!(s: Path::from_str(""), ".");
        t!(s: Path::from_str("/"), "/");
        t!(s: Path::from_str("hi"), "hi");
        t!(s: Path::from_str("hi/"), "hi");
        t!(s: Path::from_str("/lib"), "/lib");
        t!(s: Path::from_str("/lib/"), "/lib");
        t!(s: Path::from_str("hi/there"), "hi/there");
        t!(s: Path::from_str("hi/there.txt"), "hi/there.txt");

        t!(s: Path::from_str("hi/there/"), "hi/there");
        t!(s: Path::from_str("hi/../there"), "there");
        t!(s: Path::from_str("../hi/there"), "../hi/there");
        t!(s: Path::from_str("/../hi/there"), "/hi/there");
        t!(s: Path::from_str("foo/.."), ".");
        t!(s: Path::from_str("/foo/.."), "/");
        t!(s: Path::from_str("/foo/../.."), "/");
        t!(s: Path::from_str("/foo/../../bar"), "/bar");
        t!(s: Path::from_str("/./hi/./there/."), "/hi/there");
        t!(s: Path::from_str("/./hi/./there/./.."), "/hi");
        t!(s: Path::from_str("foo/../.."), "..");
        t!(s: Path::from_str("foo/../../.."), "../..");
        t!(s: Path::from_str("foo/../../bar"), "../bar");

        assert_eq!(Path::from_vec(b!("foo/bar")).into_vec(), b!("foo/bar").to_owned());
        assert_eq!(Path::from_vec(b!("/foo/../../bar")).into_vec(),
                   b!("/bar").to_owned());
        assert_eq!(Path::from_str("foo/bar").into_str(), Some(~"foo/bar"));
        assert_eq!(Path::from_str("/foo/../../bar").into_str(), Some(~"/bar"));

        let p = Path::from_vec(b!("foo/bar", 0x80));
        assert_eq!(p.as_str(), None);
        assert_eq!(Path::from_vec(b!("foo", 0xff, "/bar")).into_str(), None);
    }

    #[test]
    fn test_opt_paths() {
        assert_eq!(Path::from_vec_opt(b!("foo/bar", 0)), None);
        t!(v: Path::from_vec_opt(b!("foo/bar")).unwrap(), b!("foo/bar"));
        assert_eq!(Path::from_str_opt("foo/bar\0"), None);
        t!(s: Path::from_str_opt("foo/bar").unwrap(), "foo/bar");
    }

    #[test]
    fn test_null_byte() {
        use path2::null_byte::cond;

        let mut handled = false;
        let mut p = do cond.trap(|v| {
            handled = true;
            assert_eq!(v.as_slice(), b!("foo/bar", 0));
            (b!("/bar").to_owned())
        }).inside {
            Path::from_vec(b!("foo/bar", 0))
        };
        assert!(handled);
        assert_eq!(p.as_vec(), b!("/bar"));

        handled = false;
        do cond.trap(|v| {
            handled = true;
            assert_eq!(v.as_slice(), b!("f", 0, "o"));
            (b!("foo").to_owned())
        }).inside {
            p.set_filename(b!("f", 0, "o"))
        };
        assert!(handled);
        assert_eq!(p.as_vec(), b!("/foo"));

        handled = false;
        do cond.trap(|v| {
            handled = true;
            assert_eq!(v.as_slice(), b!("null/", 0, "/byte"));
            (b!("null/byte").to_owned())
        }).inside {
            p.set_dirname(b!("null/", 0, "/byte"));
        };
        assert!(handled);
        assert_eq!(p.as_vec(), b!("null/byte/foo"));

        handled = false;
        do cond.trap(|v| {
            handled = true;
            assert_eq!(v.as_slice(), b!("f", 0, "o"));
            (b!("foo").to_owned())
        }).inside {
            p.push(b!("f", 0, "o"));
        };
        assert!(handled);
        assert_eq!(p.as_vec(), b!("null/byte/foo/foo"));
    }

    #[test]
    fn test_null_byte_fail() {
        use path2::null_byte::cond;
        use task;

        macro_rules! t(
            ($name:expr => $code:block) => (
                {
                    let mut t = task::task();
                    t.supervised();
                    t.name($name);
                    let res = do t.try $code;
                    assert!(res.is_err());
                }
            )
        )

        t!(~"from_vec() w/nul" => {
            do cond.trap(|_| {
                (b!("null", 0).to_owned())
            }).inside {
                Path::from_vec(b!("foo/bar", 0))
            };
        })

        t!(~"set_filename w/nul" => {
            let mut p = Path::from_vec(b!("foo/bar"));
            do cond.trap(|_| {
                (b!("null", 0).to_owned())
            }).inside {
                p.set_filename(b!("foo", 0))
            };
        })

        t!(~"set_dirname w/nul" => {
            let mut p = Path::from_vec(b!("foo/bar"));
            do cond.trap(|_| {
                (b!("null", 0).to_owned())
            }).inside {
                p.set_dirname(b!("foo", 0))
            };
        })

        t!(~"push w/nul" => {
            let mut p = Path::from_vec(b!("foo/bar"));
            do cond.trap(|_| {
                (b!("null", 0).to_owned())
            }).inside {
                p.push(b!("foo", 0))
            };
        })
    }

    #[test]
    fn test_components() {
        macro_rules! t(
            (s: $path:expr, $op:ident, $exp:expr) => (
                {
                    let path = Path::from_str($path);
                    assert_eq!(path.$op(), ($exp).as_bytes());
                }
            );
            (s: $path:expr, $op:ident, $exp:expr, opt) => (
                {
                    let path = Path::from_str($path);
                    let left = path.$op().map(|&x| str::from_utf8_slice(x));
                    assert_eq!(left, $exp);
                }
            );
            (v: $path:expr, $op:ident, $exp:expr) => (
                {
                    let path = Path::from_vec($path);
                    assert_eq!(path.$op(), $exp);
                }
            )
        )

        t!(v: b!("a/b/c"), filename, b!("c"));
        t!(v: b!("a/b/c", 0xff), filename, b!("c", 0xff));
        t!(v: b!("a/b", 0xff, "/c"), filename, b!("c"));
        t!(s: "a/b/c", filename, "c");
        t!(s: "/a/b/c", filename, "c");
        t!(s: "a", filename, "a");
        t!(s: "/a", filename, "a");
        t!(s: ".", filename, "");
        t!(s: "/", filename, "");
        t!(s: "..", filename, "");
        t!(s: "../..", filename, "");

        t!(v: b!("a/b/c"), dirname, b!("a/b"));
        t!(v: b!("a/b/c", 0xff), dirname, b!("a/b"));
        t!(v: b!("a/b", 0xff, "/c"), dirname, b!("a/b", 0xff));
        t!(s: "a/b/c", dirname, "a/b");
        t!(s: "/a/b/c", dirname, "/a/b");
        t!(s: "a", dirname, ".");
        t!(s: "/a", dirname, "/");
        t!(s: ".", dirname, ".");
        t!(s: "/", dirname, "/");
        t!(s: "..", dirname, "..");
        t!(s: "../..", dirname, "../..");

        t!(v: b!("hi/there.txt"), filestem, b!("there"));
        t!(v: b!("hi/there", 0x80, ".txt"), filestem, b!("there", 0x80));
        t!(v: b!("hi/there.t", 0x80, "xt"), filestem, b!("there"));
        t!(s: "hi/there.txt", filestem, "there");
        t!(s: "hi/there", filestem, "there");
        t!(s: "there.txt", filestem, "there");
        t!(s: "there", filestem, "there");
        t!(s: ".", filestem, "");
        t!(s: "/", filestem, "");
        t!(s: "foo/.bar", filestem, ".bar");
        t!(s: ".bar", filestem, ".bar");
        t!(s: "..bar", filestem, ".");
        t!(s: "hi/there..txt", filestem, "there.");
        t!(s: "..", filestem, "");
        t!(s: "../..", filestem, "");

        t!(v: b!("hi/there.txt"), extension, Some(b!("txt")));
        t!(v: b!("hi/there", 0x80, ".txt"), extension, Some(b!("txt")));
        t!(v: b!("hi/there.t", 0x80, "xt"), extension, Some(b!("t", 0x80, "xt")));
        t!(v: b!("hi/there"), extension, None);
        t!(v: b!("hi/there", 0x80), extension, None);
        t!(s: "hi/there.txt", extension, Some("txt"), opt);
        t!(s: "hi/there", extension, None, opt);
        t!(s: "there.txt", extension, Some("txt"), opt);
        t!(s: "there", extension, None, opt);
        t!(s: ".", extension, None, opt);
        t!(s: "/", extension, None, opt);
        t!(s: "foo/.bar", extension, None, opt);
        t!(s: ".bar", extension, None, opt);
        t!(s: "..bar", extension, Some("bar"), opt);
        t!(s: "hi/there..txt", extension, Some("txt"), opt);
        t!(s: "..", extension, None, opt);
        t!(s: "../..", extension, None, opt);
    }

    #[test]
    fn test_push() {
        macro_rules! t(
            (s: $path:expr, $join:expr) => (
                {
                    let path = ($path);
                    let join = ($join);
                    let mut p1 = Path::from_str(path);
                    let p2 = p1.clone();
                    p1.push_str(join);
                    assert_eq!(p1, p2.join_str(join));
                }
            )
        )

        t!(s: "a/b/c", "..");
        t!(s: "/a/b/c", "d");
        t!(s: "a/b", "c/d");
        t!(s: "a/b", "/c/d");
    }

    #[test]
    fn test_push_path() {
        macro_rules! t(
            (s: $path:expr, $push:expr, $exp:expr) => (
                {
                    let mut p = Path::from_str($path);
                    let push = Path::from_str($push);
                    p.push_path(&push);
                    assert_eq!(p.as_str(), Some($exp));
                }
            )
        )

        t!(s: "a/b/c", "d", "a/b/c/d");
        t!(s: "/a/b/c", "d", "/a/b/c/d");
        t!(s: "a/b", "c/d", "a/b/c/d");
        t!(s: "a/b", "/c/d", "/c/d");
        t!(s: "a/b", ".", "a/b");
        t!(s: "a/b", "../c", "a/c");
    }

    #[test]
    fn test_pop() {
        macro_rules! t(
            (s: $path:expr, $left:expr, $right:expr) => (
                {
                    let mut p = Path::from_str($path);
                    let file = p.pop_opt_str();
                    assert_eq!(p.as_str(), Some($left));
                    assert_eq!(file.map(|s| s.as_slice()), $right);
                }
            );
            (v: [$($path:expr),+], [$($left:expr),+], Some($($right:expr),+)) => (
                {
                    let mut p = Path::from_vec(b!($($path),+));
                    let file = p.pop_opt();
                    assert_eq!(p.as_vec(), b!($($left),+));
                    assert_eq!(file.map(|v| v.as_slice()), Some(b!($($right),+)));
                }
            );
            (v: [$($path:expr),+], [$($left:expr),+], None) => (
                {
                    let mut p = Path::from_vec(b!($($path),+));
                    let file = p.pop_opt();
                    assert_eq!(p.as_vec(), b!($($left),+));
                    assert_eq!(file, None);
                }
            )
        )

        t!(v: ["a/b/c"], ["a/b"], Some("c"));
        t!(v: ["a"], ["."], Some("a"));
        t!(v: ["."], ["."], None);
        t!(v: ["/a"], ["/"], Some("a"));
        t!(v: ["/"], ["/"], None);
        t!(v: ["a/b/c", 0x80], ["a/b"], Some("c", 0x80));
        t!(v: ["a/b", 0x80, "/c"], ["a/b", 0x80], Some("c"));
        t!(v: [0xff], ["."], Some(0xff));
        t!(v: ["/", 0xff], ["/"], Some(0xff));
        t!(s: "a/b/c", "a/b", Some("c"));
        t!(s: "a", ".", Some("a"));
        t!(s: ".", ".", None);
        t!(s: "/a", "/", Some("a"));
        t!(s: "/", "/", None);

        assert_eq!(Path::from_vec(b!("foo/bar", 0x80)).pop_opt_str(), None);
        assert_eq!(Path::from_vec(b!("foo", 0x80, "/bar")).pop_opt_str(), Some(~"bar"));
    }

    #[test]
    fn test_join() {
        t!(v: Path::from_vec(b!("a/b/c")).join(b!("..")), b!("a/b"));
        t!(v: Path::from_vec(b!("/a/b/c")).join(b!("d")), b!("/a/b/c/d"));
        t!(v: Path::from_vec(b!("a/", 0x80, "/c")).join(b!(0xff)), b!("a/", 0x80, "/c/", 0xff));
        t!(s: Path::from_str("a/b/c").join_str(".."), "a/b");
        t!(s: Path::from_str("/a/b/c").join_str("d"), "/a/b/c/d");
        t!(s: Path::from_str("a/b").join_str("c/d"), "a/b/c/d");
        t!(s: Path::from_str("a/b").join_str("/c/d"), "/c/d");
        t!(s: Path::from_str(".").join_str("a/b"), "a/b");
        t!(s: Path::from_str("/").join_str("a/b"), "/a/b");
    }

    #[test]
    fn test_join_path() {
        macro_rules! t(
            (s: $path:expr, $join:expr, $exp:expr) => (
                {
                    let path = Path::from_str($path);
                    let join = Path::from_str($join);
                    let res = path.join_path(&join);
                    assert_eq!(res.as_str(), Some($exp));
                }
            )
        )

        t!(s: "a/b/c", "..", "a/b");
        t!(s: "/a/b/c", "d", "/a/b/c/d");
        t!(s: "a/b", "c/d", "a/b/c/d");
        t!(s: "a/b", "/c/d", "/c/d");
        t!(s: ".", "a/b", "a/b");
        t!(s: "/", "a/b", "/a/b");
    }

    #[test]
    fn test_with_helpers() {
        t!(v: Path::from_vec(b!("a/b/c")).with_dirname(b!("d")), b!("d/c"));
        t!(v: Path::from_vec(b!("a/b/c")).with_dirname(b!("d/e")), b!("d/e/c"));
        t!(v: Path::from_vec(b!("a/", 0x80, "b/c")).with_dirname(b!(0xff)), b!(0xff, "/c"));
        t!(v: Path::from_vec(b!("a/b/", 0x80)).with_dirname(b!("/", 0xcd)),
              b!("/", 0xcd, "/", 0x80));
        t!(s: Path::from_str("a/b/c").with_dirname_str("d"), "d/c");
        t!(s: Path::from_str("a/b/c").with_dirname_str("d/e"), "d/e/c");
        t!(s: Path::from_str("a/b/c").with_dirname_str(""), "c");
        t!(s: Path::from_str("a/b/c").with_dirname_str("/"), "/c");
        t!(s: Path::from_str("a/b/c").with_dirname_str("."), "c");
        t!(s: Path::from_str("a/b/c").with_dirname_str(".."), "../c");
        t!(s: Path::from_str("/").with_dirname_str("foo"), "foo");
        t!(s: Path::from_str("/").with_dirname_str(""), ".");
        t!(s: Path::from_str("/foo").with_dirname_str("bar"), "bar/foo");
        t!(s: Path::from_str("..").with_dirname_str("foo"), "foo");
        t!(s: Path::from_str("../..").with_dirname_str("foo"), "foo");
        t!(s: Path::from_str("..").with_dirname_str(""), ".");
        t!(s: Path::from_str("../..").with_dirname_str(""), ".");
        t!(s: Path::from_str("foo").with_dirname_str(".."), "../foo");
        t!(s: Path::from_str("foo").with_dirname_str("../.."), "../../foo");

        t!(v: Path::from_vec(b!("a/b/c")).with_filename(b!("d")), b!("a/b/d"));
        t!(v: Path::from_vec(b!("a/b/c", 0xff)).with_filename(b!(0x80)), b!("a/b/", 0x80));
        t!(v: Path::from_vec(b!("/", 0xff, "/foo")).with_filename(b!(0xcd)),
              b!("/", 0xff, "/", 0xcd));
        t!(s: Path::from_str("a/b/c").with_filename_str("d"), "a/b/d");
        t!(s: Path::from_str(".").with_filename_str("foo"), "foo");
        t!(s: Path::from_str("/a/b/c").with_filename_str("d"), "/a/b/d");
        t!(s: Path::from_str("/").with_filename_str("foo"), "/foo");
        t!(s: Path::from_str("/a").with_filename_str("foo"), "/foo");
        t!(s: Path::from_str("foo").with_filename_str("bar"), "bar");
        t!(s: Path::from_str("/").with_filename_str("foo/"), "/foo");
        t!(s: Path::from_str("/a").with_filename_str("foo/"), "/foo");
        t!(s: Path::from_str("a/b/c").with_filename_str(""), "a/b");
        t!(s: Path::from_str("a/b/c").with_filename_str("."), "a/b");
        t!(s: Path::from_str("a/b/c").with_filename_str(".."), "a");
        t!(s: Path::from_str("/a").with_filename_str(""), "/");
        t!(s: Path::from_str("foo").with_filename_str(""), ".");
        t!(s: Path::from_str("a/b/c").with_filename_str("d/e"), "a/b/d/e");
        t!(s: Path::from_str("a/b/c").with_filename_str("/d"), "a/b/d");
        t!(s: Path::from_str("..").with_filename_str("foo"), "../foo");
        t!(s: Path::from_str("../..").with_filename_str("foo"), "../../foo");
        t!(s: Path::from_str("..").with_filename_str(""), "..");
        t!(s: Path::from_str("../..").with_filename_str(""), "../..");

        t!(v: Path::from_vec(b!("hi/there", 0x80, ".txt")).with_filestem(b!(0xff)),
              b!("hi/", 0xff, ".txt"));
        t!(v: Path::from_vec(b!("hi/there.txt", 0x80)).with_filestem(b!(0xff)),
              b!("hi/", 0xff, ".txt", 0x80));
        t!(v: Path::from_vec(b!("hi/there", 0xff)).with_filestem(b!(0x80)), b!("hi/", 0x80));
        t!(v: Path::from_vec(b!("hi", 0x80, "/there")).with_filestem([]), b!("hi", 0x80));
        t!(s: Path::from_str("hi/there.txt").with_filestem_str("here"), "hi/here.txt");
        t!(s: Path::from_str("hi/there.txt").with_filestem_str(""), "hi/.txt");
        t!(s: Path::from_str("hi/there.txt").with_filestem_str("."), "hi/..txt");
        t!(s: Path::from_str("hi/there.txt").with_filestem_str(".."), "hi/...txt");
        t!(s: Path::from_str("hi/there.txt").with_filestem_str("/"), "hi/.txt");
        t!(s: Path::from_str("hi/there.txt").with_filestem_str("foo/bar"), "hi/foo/bar.txt");
        t!(s: Path::from_str("hi/there.foo.txt").with_filestem_str("here"), "hi/here.txt");
        t!(s: Path::from_str("hi/there").with_filestem_str("here"), "hi/here");
        t!(s: Path::from_str("hi/there").with_filestem_str(""), "hi");
        t!(s: Path::from_str("hi").with_filestem_str(""), ".");
        t!(s: Path::from_str("/hi").with_filestem_str(""), "/");
        t!(s: Path::from_str("hi/there").with_filestem_str(".."), ".");
        t!(s: Path::from_str("hi/there").with_filestem_str("."), "hi");
        t!(s: Path::from_str("hi/there.").with_filestem_str("foo"), "hi/foo.");
        t!(s: Path::from_str("hi/there.").with_filestem_str(""), "hi");
        t!(s: Path::from_str("hi/there.").with_filestem_str("."), ".");
        t!(s: Path::from_str("hi/there.").with_filestem_str(".."), "hi/...");
        t!(s: Path::from_str("/").with_filestem_str("foo"), "/foo");
        t!(s: Path::from_str(".").with_filestem_str("foo"), "foo");
        t!(s: Path::from_str("hi/there..").with_filestem_str("here"), "hi/here.");
        t!(s: Path::from_str("hi/there..").with_filestem_str(""), "hi");

        t!(v: Path::from_vec(b!("hi/there", 0x80, ".txt")).with_extension(b!("exe")),
              b!("hi/there", 0x80, ".exe"));
        t!(v: Path::from_vec(b!("hi/there.txt", 0x80)).with_extension(b!(0xff)),
              b!("hi/there.", 0xff));
        t!(v: Path::from_vec(b!("hi/there", 0x80)).with_extension(b!(0xff)),
              b!("hi/there", 0x80, ".", 0xff));
        t!(v: Path::from_vec(b!("hi/there.", 0xff)).with_extension([]), b!("hi/there"));
        t!(s: Path::from_str("hi/there.txt").with_extension_str("exe"), "hi/there.exe");
        t!(s: Path::from_str("hi/there.txt").with_extension_str(""), "hi/there");
        t!(s: Path::from_str("hi/there.txt").with_extension_str("."), "hi/there..");
        t!(s: Path::from_str("hi/there.txt").with_extension_str(".."), "hi/there...");
        t!(s: Path::from_str("hi/there").with_extension_str("txt"), "hi/there.txt");
        t!(s: Path::from_str("hi/there").with_extension_str("."), "hi/there..");
        t!(s: Path::from_str("hi/there").with_extension_str(".."), "hi/there...");
        t!(s: Path::from_str("hi/there.").with_extension_str("txt"), "hi/there.txt");
        t!(s: Path::from_str("hi/.foo").with_extension_str("txt"), "hi/.foo.txt");
        t!(s: Path::from_str("hi/there.txt").with_extension_str(".foo"), "hi/there..foo");
        t!(s: Path::from_str("/").with_extension_str("txt"), "/");
        t!(s: Path::from_str("/").with_extension_str("."), "/");
        t!(s: Path::from_str("/").with_extension_str(".."), "/");
        t!(s: Path::from_str(".").with_extension_str("txt"), ".");
    }

    #[test]
    fn test_setters() {
        macro_rules! t(
            (s: $path:expr, $set:ident, $with:ident, $arg:expr) => (
                {
                    let path = $path;
                    let arg = $arg;
                    let mut p1 = Path::from_str(path);
                    p1.$set(arg);
                    let p2 = Path::from_str(path);
                    assert_eq!(p1, p2.$with(arg));
                }
            );
            (v: $path:expr, $set:ident, $with:ident, $arg:expr) => (
                {
                    let path = $path;
                    let arg = $arg;
                    let mut p1 = Path::from_vec(path);
                    p1.$set(arg);
                    let p2 = Path::from_vec(path);
                    assert_eq!(p1, p2.$with(arg));
                }
            )
        )

        t!(v: b!("a/b/c"), set_dirname, with_dirname, b!("d"));
        t!(v: b!("a/b/c"), set_dirname, with_dirname, b!("d/e"));
        t!(v: b!("a/", 0x80, "/c"), set_dirname, with_dirname, b!(0xff));
        t!(s: "a/b/c", set_dirname_str, with_dirname_str, "d");
        t!(s: "a/b/c", set_dirname_str, with_dirname_str, "d/e");
        t!(s: "/", set_dirname_str, with_dirname_str, "foo");
        t!(s: "/foo", set_dirname_str, with_dirname_str, "bar");
        t!(s: "a/b/c", set_dirname_str, with_dirname_str, "");
        t!(s: "../..", set_dirname_str, with_dirname_str, "x");
        t!(s: "foo", set_dirname_str, with_dirname_str, "../..");

        t!(v: b!("a/b/c"), set_filename, with_filename, b!("d"));
        t!(v: b!("/"), set_filename, with_filename, b!("foo"));
        t!(v: b!(0x80), set_filename, with_filename, b!(0xff));
        t!(s: "a/b/c", set_filename_str, with_filename_str, "d");
        t!(s: "/", set_filename_str, with_filename_str, "foo");
        t!(s: ".", set_filename_str, with_filename_str, "foo");
        t!(s: "a/b", set_filename_str, with_filename_str, "");
        t!(s: "a", set_filename_str, with_filename_str, "");

        t!(v: b!("hi/there.txt"), set_filestem, with_filestem, b!("here"));
        t!(v: b!("hi/there", 0x80, ".txt"), set_filestem, with_filestem, b!("here", 0xff));
        t!(s: "hi/there.txt", set_filestem_str, with_filestem_str, "here");
        t!(s: "hi/there.", set_filestem_str, with_filestem_str, "here");
        t!(s: "hi/there", set_filestem_str, with_filestem_str, "here");
        t!(s: "hi/there.txt", set_filestem_str, with_filestem_str, "");
        t!(s: "hi/there", set_filestem_str, with_filestem_str, "");

        t!(v: b!("hi/there.txt"), set_extension, with_extension, b!("exe"));
        t!(v: b!("hi/there.t", 0x80, "xt"), set_extension, with_extension, b!("exe", 0xff));
        t!(s: "hi/there.txt", set_extension_str, with_extension_str, "exe");
        t!(s: "hi/there.", set_extension_str, with_extension_str, "txt");
        t!(s: "hi/there", set_extension_str, with_extension_str, "txt");
        t!(s: "hi/there.txt", set_extension_str, with_extension_str, "");
        t!(s: "hi/there", set_extension_str, with_extension_str, "");
        t!(s: ".", set_extension_str, with_extension_str, "txt");
    }

    #[test]
    fn test_getters() {
        macro_rules! t(
            (s: $path:expr, $filename:expr, $dirname:expr, $filestem:expr, $ext:expr) => (
                {
                    let path = $path;
                    assert_eq!(path.filename_str(), $filename);
                    assert_eq!(path.dirname_str(), $dirname);
                    assert_eq!(path.filestem_str(), $filestem);
                    assert_eq!(path.extension_str(), $ext);
                }
            );
            (v: $path:expr, $filename:expr, $dirname:expr, $filestem:expr, $ext:expr) => (
                {
                    let path = $path;
                    assert_eq!(path.filename(), $filename);
                    assert_eq!(path.dirname(), $dirname);
                    assert_eq!(path.filestem(), $filestem);
                    assert_eq!(path.extension(), $ext);
                }
            )
        )

        t!(v: Path::from_vec(b!("a/b/c")), b!("c"), b!("a/b"), b!("c"), None);
        t!(v: Path::from_vec(b!("a/b/", 0xff)), b!(0xff), b!("a/b"), b!(0xff), None);
        t!(v: Path::from_vec(b!("hi/there.", 0xff)), b!("there.", 0xff), b!("hi"),
              b!("there"), Some(b!(0xff)));
        t!(s: Path::from_str("a/b/c"), Some("c"), Some("a/b"), Some("c"), None);
        t!(s: Path::from_str("."), Some(""), Some("."), Some(""), None);
        t!(s: Path::from_str("/"), Some(""), Some("/"), Some(""), None);
        t!(s: Path::from_str(".."), Some(""), Some(".."), Some(""), None);
        t!(s: Path::from_str("../.."), Some(""), Some("../.."), Some(""), None);
        t!(s: Path::from_str("hi/there.txt"), Some("there.txt"), Some("hi"),
              Some("there"), Some("txt"));
        t!(s: Path::from_str("hi/there"), Some("there"), Some("hi"), Some("there"), None);
        t!(s: Path::from_str("hi/there."), Some("there."), Some("hi"),
              Some("there"), Some(""));
        t!(s: Path::from_str("hi/.there"), Some(".there"), Some("hi"), Some(".there"), None);
        t!(s: Path::from_str("hi/..there"), Some("..there"), Some("hi"),
              Some("."), Some("there"));
        t!(s: Path::from_vec(b!("a/b/", 0xff)), None, Some("a/b"), None, None);
        t!(s: Path::from_vec(b!("a/b/", 0xff, ".txt")), None, Some("a/b"), None, Some("txt"));
        t!(s: Path::from_vec(b!("a/b/c.", 0x80)), None, Some("a/b"), Some("c"), None);
        t!(s: Path::from_vec(b!(0xff, "/b")), Some("b"), None, Some("b"), None);
    }

    #[test]
    fn test_dir_file_path() {
        t!(v: Path::from_vec(b!("hi/there", 0x80)).dir_path(), b!("hi"));
        t!(v: Path::from_vec(b!("hi", 0xff, "/there")).dir_path(), b!("hi", 0xff));
        t!(s: Path::from_str("hi/there").dir_path(), "hi");
        t!(s: Path::from_str("hi").dir_path(), ".");
        t!(s: Path::from_str("/hi").dir_path(), "/");
        t!(s: Path::from_str("/").dir_path(), "/");
        t!(s: Path::from_str("..").dir_path(), "..");
        t!(s: Path::from_str("../..").dir_path(), "../..");

        macro_rules! t(
            (s: $path:expr, $exp:expr) => (
                {
                    let path = $path;
                    let left = path.and_then_ref(|p| p.as_str());
                    assert_eq!(left, $exp);
                }
            );
            (v: $path:expr, $exp:expr) => (
                {
                    let path = $path;
                    let left = path.map(|p| p.as_vec());
                    assert_eq!(left, $exp);
                }
            )
        )

        t!(v: Path::from_vec(b!("hi/there", 0x80)).file_path(), Some(b!("there", 0x80)));
        t!(v: Path::from_vec(b!("hi", 0xff, "/there")).file_path(), Some(b!("there")));
        t!(s: Path::from_str("hi/there").file_path(), Some("there"));
        t!(s: Path::from_str("hi").file_path(), Some("hi"));
        t!(s: Path::from_str(".").file_path(), None);
        t!(s: Path::from_str("/").file_path(), None);
        t!(s: Path::from_str("..").file_path(), None);
        t!(s: Path::from_str("../..").file_path(), None);
    }

    #[test]
    fn test_is_absolute() {
        assert_eq!(Path::from_str("a/b/c").is_absolute(), false);
        assert_eq!(Path::from_str("/a/b/c").is_absolute(), true);
        assert_eq!(Path::from_str("a").is_absolute(), false);
        assert_eq!(Path::from_str("/a").is_absolute(), true);
        assert_eq!(Path::from_str(".").is_absolute(), false);
        assert_eq!(Path::from_str("/").is_absolute(), true);
        assert_eq!(Path::from_str("..").is_absolute(), false);
        assert_eq!(Path::from_str("../..").is_absolute(), false);
    }

    #[test]
    fn test_is_ancestor_of() {
        macro_rules! t(
            (s: $path:expr, $dest:expr, $exp:expr) => (
                {
                    let path = Path::from_str($path);
                    let dest = Path::from_str($dest);
                    assert_eq!(path.is_ancestor_of(&dest), $exp);
                }
            )
        )

        t!(s: "a/b/c", "a/b/c/d", true);
        t!(s: "a/b/c", "a/b/c", true);
        t!(s: "a/b/c", "a/b", false);
        t!(s: "/a/b/c", "/a/b/c", true);
        t!(s: "/a/b", "/a/b/c", true);
        t!(s: "/a/b/c/d", "/a/b/c", false);
        t!(s: "/a/b", "a/b/c", false);
        t!(s: "a/b", "/a/b/c", false);
        t!(s: "a/b/c", "a/b/d", false);
        t!(s: "../a/b/c", "a/b/c", false);
        t!(s: "a/b/c", "../a/b/c", false);
        t!(s: "a/b/c", "a/b/cd", false);
        t!(s: "a/b/cd", "a/b/c", false);
        t!(s: "../a/b", "../a/b/c", true);
        t!(s: ".", "a/b", true);
        t!(s: ".", ".", true);
        t!(s: "/", "/", true);
        t!(s: "/", "/a/b", true);
        t!(s: "..", "a/b", true);
        t!(s: "../..", "a/b", true);
    }

    #[test]
    fn test_path_relative_from() {
        macro_rules! t(
            (s: $path:expr, $other:expr, $exp:expr) => (
                {
                    let path = Path::from_str($path);
                    let other = Path::from_str($other);
                    let res = path.path_relative_from(&other);
                    assert_eq!(res.and_then_ref(|x| x.as_str()), $exp);
                }
            )
        )

        t!(s: "a/b/c", "a/b", Some("c"));
        t!(s: "a/b/c", "a/b/d", Some("../c"));
        t!(s: "a/b/c", "a/b/c/d", Some(".."));
        t!(s: "a/b/c", "a/b/c", Some("."));
        t!(s: "a/b/c", "a/b/c/d/e", Some("../.."));
        t!(s: "a/b/c", "a/d/e", Some("../../b/c"));
        t!(s: "a/b/c", "d/e/f", Some("../../../a/b/c"));
        t!(s: "a/b/c", "/a/b/c", None);
        t!(s: "/a/b/c", "a/b/c", Some("/a/b/c"));
        t!(s: "/a/b/c", "/a/b/c/d", Some(".."));
        t!(s: "/a/b/c", "/a/b", Some("c"));
        t!(s: "/a/b/c", "/a/b/c/d/e", Some("../.."));
        t!(s: "/a/b/c", "/a/d/e", Some("../../b/c"));
        t!(s: "/a/b/c", "/d/e/f", Some("../../../a/b/c"));
        t!(s: "hi/there.txt", "hi/there", Some("../there.txt"));
        t!(s: ".", "a", Some(".."));
        t!(s: ".", "a/b", Some("../.."));
        t!(s: ".", ".", Some("."));
        t!(s: "a", ".", Some("a"));
        t!(s: "a/b", ".", Some("a/b"));
        t!(s: "..", ".", Some(".."));
        t!(s: "a/b/c", "a/b/c", Some("."));
        t!(s: "/a/b/c", "/a/b/c", Some("."));
        t!(s: "/", "/", Some("."));
        t!(s: "/", ".", Some("/"));
        t!(s: "../../a", "b", Some("../../../a"));
        t!(s: "a", "../../b", None);
        t!(s: "../../a", "../../b", Some("../a"));
        t!(s: "../../a", "../../a/b", Some(".."));
        t!(s: "../../a/b", "../../a", Some("b"));
    }

    #[test]
    fn test_component_iter() {
        macro_rules! t(
            (s: $path:expr, $exp:expr) => (
                {
                    let path = Path::from_str($path);
                    let comps = path.component_iter().to_owned_vec();
                    let exp: &[&str] = $exp;
                    let exps = exp.iter().map(|x| x.as_bytes()).to_owned_vec();
                    assert_eq!(comps, exps);
                }
            );
            (v: [$($arg:expr),+], [$([$($exp:expr),*]),*]) => (
                {
                    let path = Path::from_vec(b!($($arg),+));
                    let comps = path.component_iter().to_owned_vec();
                    let exp: &[&[u8]] = [$(b!($($exp),*)),*];
                    assert_eq!(comps.as_slice(), exp);
                }
            )
        )

        t!(v: ["a/b/c"], [["a"], ["b"], ["c"]]);
        t!(v: ["/", 0xff, "/a/", 0x80], [[0xff], ["a"], [0x80]]);
        t!(v: ["../../foo", 0xcd, "bar"], [[".."], [".."], ["foo", 0xcd, "bar"]]);
        t!(s: "a/b/c", ["a", "b", "c"]);
        t!(s: "a/b/d", ["a", "b", "d"]);
        t!(s: "a/b/cd", ["a", "b", "cd"]);
        t!(s: "/a/b/c", ["a", "b", "c"]);
        t!(s: "a", ["a"]);
        t!(s: "/a", ["a"]);
        t!(s: "/", []);
        t!(s: ".", ["."]);
        t!(s: "..", [".."]);
        t!(s: "../..", ["..", ".."]);
        t!(s: "../../foo", ["..", "..", "foo"]);
    }
}
