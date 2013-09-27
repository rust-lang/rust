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
use iter::{AdditiveIterator, Extendable, Iterator, Map};
use option::{Option, None, Some};
use str;
use str::Str;
use to_bytes::IterBytes;
use util;
use vec;
use vec::{CopyableVector, RSplitIterator, SplitIterator, Vector, VectorVector};
use super::{GenericPath, GenericPathUnsafe};

#[cfg(not(target_os = "win32"))]
use libc;

/// Iterator that yields successive components of a Path as &[u8]
pub type ComponentIter<'self> = SplitIterator<'self, u8>;
/// Iterator that yields components of a Path in reverse as &[u8]
pub type RevComponentIter<'self> = RSplitIterator<'self, u8>;

/// Iterator that yields successive components of a Path as Option<&str>
pub type StrComponentIter<'self> = Map<'self, &'self [u8], Option<&'self str>,
                                       ComponentIter<'self>>;
/// Iterator that yields components of a Path in reverse as Option<&str>
pub type RevStrComponentIter<'self> = Map<'self, &'self [u8], Option<&'self str>,
                                          RevComponentIter<'self>>;

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

impl IterBytes for Path {
    #[inline]
    fn iter_bytes(&self, lsb0: bool, f: &fn(buf: &[u8]) -> bool) -> bool {
        self.repr.iter_bytes(lsb0, f)
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

    fn filename<'a>(&'a self) -> Option<&'a [u8]> {
        match self.sepidx {
            None if bytes!(".") == self.repr || bytes!("..") == self.repr => None,
            None => Some(self.repr.as_slice()),
            Some(idx) if self.repr.slice_from(idx+1) == bytes!("..") => None,
            Some(0) if self.repr.slice_from(1).is_empty() => None,
            Some(idx) => Some(self.repr.slice_from(idx+1))
        }
    }

    fn pop(&mut self) -> Option<~[u8]> {
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

    fn root_path(&self) -> Option<Path> {
        if self.is_absolute() {
            Some(Path::from_str("/"))
        } else {
            None
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

    /// Returns an iterator that yields each component of the path in reverse.
    /// See component_iter() for details.
    pub fn rev_component_iter<'a>(&'a self) -> RevComponentIter<'a> {
        let v = if self.repr[0] == sep {
            self.repr.slice_from(1)
        } else { self.repr.as_slice() };
        let mut ret = v.rsplit_iter(is_sep);
        if v.is_empty() {
            // consume the empty "" component
            ret.next();
        }
        ret
    }

    /// Returns an iterator that yields each component of the path as Option<&str>.
    /// See component_iter() for details.
    pub fn str_component_iter<'a>(&'a self) -> StrComponentIter<'a> {
        self.component_iter().map(str::from_utf8_slice_opt)
    }

    /// Returns an iterator that yields each component of the path in reverse as Option<&str>.
    /// See component_iter() for details.
    pub fn rev_str_component_iter<'a>(&'a self) -> RevStrComponentIter<'a> {
        self.rev_component_iter().map(str::from_utf8_slice_opt)
    }

    /// Returns whether the relative path `child` is a suffix of `self`.
    pub fn ends_with_path(&self, child: &Path) -> bool {
        if !child.is_relative() { return false; }
        let mut selfit = self.rev_component_iter();
        let mut childit = child.rev_component_iter();
        loop {
            match (selfit.next(), childit.next()) {
                (Some(a), Some(b)) => if a != b { return false; },
                (Some(_), None) => break,
                (None, Some(_)) => return false,
                (None, None) => break
            }
        }
        true
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
            else { comps.pop(); changed = true }
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

static dot_static: &'static [u8] = bytes!(".");
static dot_dot_static: &'static [u8] = bytes!("..");

// Stat support
#[cfg(not(target_os = "win32"))]
impl Path {
    /// Calls stat() on the represented file and returns the resulting libc::stat
    pub fn stat(&self) -> Option<libc::stat> {
        #[fixed_stack_segment]; #[inline(never)];
        do self.with_c_str |buf| {
            let mut st = super::stat::arch::default_stat();
            match unsafe { libc::stat(buf as *libc::c_char, &mut st) } {
                0 => Some(st),
                _ => None
            }
        }
    }

    /// Returns whether the represented file exists
    pub fn exists(&self) -> bool {
        match self.stat() {
            None => false,
            Some(_) => true
        }
    }

    /// Returns the filesize of the represented file
    pub fn get_size(&self) -> Option<i64> {
        match self.stat() {
            None => None,
            Some(st) => Some(st.st_size as i64)
        }
    }

    /// Returns the mode of the represented file
    pub fn get_mode(&self) -> Option<uint> {
        match self.stat() {
            None => None,
            Some(st) => Some(st.st_mode as uint)
        }
    }
}

#[cfg(target_os = "freebsd")]
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
impl Path {
    /// Returns the atime of the represented file, as (secs, nsecs)
    pub fn get_atime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(st) => Some((st.st_atime as i64, st.st_atime_nsec as int))
        }
    }

    /// Returns the mtime of the represented file, as (secs, nsecs)
    pub fn get_mtime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(st) => Some((st.st_mtime as i64, st.st_mtime_nsec as int))
        }
    }

    /// Returns the ctime of the represented file, as (secs, nsecs)
    pub fn get_ctime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(st) => Some((st.st_ctime as i64, st.st_ctime_nsec as int))
        }
    }
}

#[cfg(unix)]
impl Path {
    /// Calls lstat() on the represented file and returns the resulting libc::stat
    pub fn lstat(&self) -> Option<libc::stat> {
        #[fixed_stack_segment]; #[inline(never)];
        do self.with_c_str |buf| {
            let mut st = super::stat::arch::default_stat();
            match unsafe { libc::lstat(buf, &mut st) } {
                0 => Some(st),
                _ => None
            }
        }
    }
}

#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
impl Path {
    /// Returns the birthtime of the represented file
    pub fn get_birthtime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(st) => Some((st.st_birthtime as i64, st.st_birthtime_nsec as int))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use option::{Option, Some, None};
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
        use path::null_byte::cond;

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
        use path::null_byte::cond;
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
    fn test_display_str() {
        assert_eq!(Path::from_str("foo").to_display_str(), ~"foo");
        assert_eq!(Path::from_vec(b!("foo", 0x80)).to_display_str(), ~"foo\uFFFD");
        assert_eq!(Path::from_vec(b!("foo", 0xff, "bar")).to_display_str(), ~"foo\uFFFDbar");
        assert_eq!(Path::from_vec(b!("foo", 0xff, "/bar")).to_filename_display_str(), Some(~"bar"));
        assert_eq!(Path::from_vec(b!("foo/", 0xff, "bar")).to_filename_display_str(),
                   Some(~"\uFFFDbar"));
        assert_eq!(Path::from_vec(b!("/")).to_filename_display_str(), None);

        let mut called = false;
        do Path::from_str("foo").with_display_str |s| {
            assert_eq!(s, "foo");
            called = true;
        };
        assert!(called);
        called = false;
        do Path::from_vec(b!("foo", 0x80)).with_display_str |s| {
            assert_eq!(s, "foo\uFFFD");
            called = true;
        };
        assert!(called);
        called = false;
        do Path::from_vec(b!("foo", 0xff, "bar")).with_display_str |s| {
            assert_eq!(s, "foo\uFFFDbar");
            called = true;
        };
        assert!(called);
        called = false;
        do Path::from_vec(b!("foo", 0xff, "/bar")).with_filename_display_str |s| {
            assert_eq!(s, Some("bar"));
            called = true;
        }
        assert!(called);
        called = false;
        do Path::from_vec(b!("foo/", 0xff, "bar")).with_filename_display_str |s| {
            assert_eq!(s, Some("\uFFFDbar"));
            called = true;
        }
        assert!(called);
        called = false;
        do Path::from_vec(b!("/")).with_filename_display_str |s| {
            assert!(s.is_none());
            called = true;
        }
        assert!(called);
    }

    #[test]
    fn test_display() {
        macro_rules! t(
            ($path:expr, $exp:expr, $expf:expr) => (
                {
                    let path = Path::from_vec($path);
                    let f = format!("{}", path.display());
                    assert_eq!(f.as_slice(), $exp);
                    let f = format!("{}", path.filename_display());
                    assert_eq!(f.as_slice(), $expf);
                }
            )
        )

        t!(b!("foo"), "foo", "foo");
        t!(b!("foo/bar"), "foo/bar", "bar");
        t!(b!("/"), "/", "");
        t!(b!("foo", 0xff), "foo\uFFFD", "foo\uFFFD");
        t!(b!("foo", 0xff, "/bar"), "foo\uFFFD/bar", "bar");
        t!(b!("foo/", 0xff, "bar"), "foo/\uFFFDbar", "\uFFFDbar");
        t!(b!(0xff, "foo/bar", 0xff), "\uFFFDfoo/bar\uFFFD", "bar\uFFFD");
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
            );
        )

        t!(v: b!("a/b/c"), filename, Some(b!("c")));
        t!(v: b!("a/b/c", 0xff), filename, Some(b!("c", 0xff)));
        t!(v: b!("a/b", 0xff, "/c"), filename, Some(b!("c")));
        t!(s: "a/b/c", filename, Some("c"), opt);
        t!(s: "/a/b/c", filename, Some("c"), opt);
        t!(s: "a", filename, Some("a"), opt);
        t!(s: "/a", filename, Some("a"), opt);
        t!(s: ".", filename, None, opt);
        t!(s: "/", filename, None, opt);
        t!(s: "..", filename, None, opt);
        t!(s: "../..", filename, None, opt);

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

        t!(v: b!("hi/there.txt"), filestem, Some(b!("there")));
        t!(v: b!("hi/there", 0x80, ".txt"), filestem, Some(b!("there", 0x80)));
        t!(v: b!("hi/there.t", 0x80, "xt"), filestem, Some(b!("there")));
        t!(s: "hi/there.txt", filestem, Some("there"), opt);
        t!(s: "hi/there", filestem, Some("there"), opt);
        t!(s: "there.txt", filestem, Some("there"), opt);
        t!(s: "there", filestem, Some("there"), opt);
        t!(s: ".", filestem, None, opt);
        t!(s: "/", filestem, None, opt);
        t!(s: "foo/.bar", filestem, Some(".bar"), opt);
        t!(s: ".bar", filestem, Some(".bar"), opt);
        t!(s: "..bar", filestem, Some("."), opt);
        t!(s: "hi/there..txt", filestem, Some("there."), opt);
        t!(s: "..", filestem, None, opt);
        t!(s: "../..", filestem, None, opt);

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
    fn test_push_many() {
        use to_man = at_vec::to_managed_move;

        macro_rules! t(
            (s: $path:expr, $push:expr, $exp:expr) => (
                {
                    let mut p = Path::from_str($path);
                    p.push_many_str($push);
                    assert_eq!(p.as_str(), Some($exp));
                }
            );
            (v: $path:expr, $push:expr, $exp:expr) => (
                {
                    let mut p = Path::from_vec($path);
                    p.push_many($push);
                    assert_eq!(p.as_vec(), $exp);
                }
            )
        )

        t!(s: "a/b/c", ["d", "e"], "a/b/c/d/e");
        t!(s: "a/b/c", ["d", "/e"], "/e");
        t!(s: "a/b/c", ["d", "/e", "f"], "/e/f");
        t!(s: "a/b/c", [~"d", ~"e"], "a/b/c/d/e");
        t!(s: "a/b/c", [@"d", @"e"], "a/b/c/d/e");
        t!(v: b!("a/b/c"), [b!("d"), b!("e")], b!("a/b/c/d/e"));
        t!(v: b!("a/b/c"), [b!("d"), b!("/e"), b!("f")], b!("/e/f"));
        t!(v: b!("a/b/c"), [b!("d").to_owned(), b!("e").to_owned()], b!("a/b/c/d/e"));
        t!(v: b!("a/b/c"), [to_man(b!("d").to_owned()), to_man(b!("e").to_owned())],
              b!("a/b/c/d/e"));
    }

    #[test]
    fn test_pop() {
        macro_rules! t(
            (s: $path:expr, $left:expr, $right:expr) => (
                {
                    let mut p = Path::from_str($path);
                    let file = p.pop_str();
                    assert_eq!(p.as_str(), Some($left));
                    assert_eq!(file.map(|s| s.as_slice()), $right);
                }
            );
            (v: [$($path:expr),+], [$($left:expr),+], Some($($right:expr),+)) => (
                {
                    let mut p = Path::from_vec(b!($($path),+));
                    let file = p.pop();
                    assert_eq!(p.as_vec(), b!($($left),+));
                    assert_eq!(file.map(|v| v.as_slice()), Some(b!($($right),+)));
                }
            );
            (v: [$($path:expr),+], [$($left:expr),+], None) => (
                {
                    let mut p = Path::from_vec(b!($($path),+));
                    let file = p.pop();
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

        assert_eq!(Path::from_vec(b!("foo/bar", 0x80)).pop_str(), None);
        assert_eq!(Path::from_vec(b!("foo", 0x80, "/bar")).pop_str(), Some(~"bar"));
    }

    #[test]
    fn test_root_path() {
        assert_eq!(Path::from_vec(b!("a/b/c")).root_path(), None);
        assert_eq!(Path::from_vec(b!("/a/b/c")).root_path(), Some(Path::from_str("/")));
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
    fn test_join_many() {
        use to_man = at_vec::to_managed_move;

        macro_rules! t(
            (s: $path:expr, $join:expr, $exp:expr) => (
                {
                    let path = Path::from_str($path);
                    let res = path.join_many_str($join);
                    assert_eq!(res.as_str(), Some($exp));
                }
            );
            (v: $path:expr, $join:expr, $exp:expr) => (
                {
                    let path = Path::from_vec($path);
                    let res = path.join_many($join);
                    assert_eq!(res.as_vec(), $exp);
                }
            )
        )

        t!(s: "a/b/c", ["d", "e"], "a/b/c/d/e");
        t!(s: "a/b/c", ["..", "d"], "a/b/d");
        t!(s: "a/b/c", ["d", "/e", "f"], "/e/f");
        t!(s: "a/b/c", [~"d", ~"e"], "a/b/c/d/e");
        t!(s: "a/b/c", [@"d", @"e"], "a/b/c/d/e");
        t!(v: b!("a/b/c"), [b!("d"), b!("e")], b!("a/b/c/d/e"));
        t!(v: b!("a/b/c"), [b!("d").to_owned(), b!("e").to_owned()], b!("a/b/c/d/e"));
        t!(v: b!("a/b/c"), [to_man(b!("d").to_owned()), to_man(b!("e").to_owned())],
              b!("a/b/c/d/e"));
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
    fn test_add_extension() {
        macro_rules! t(
            (s: $path:expr, $ext:expr, $exp:expr) => (
                {
                    let mut path = Path::from_str($path);
                    path.add_extension_str($ext);
                    assert_eq!(path.as_str(), Some($exp));
                }
            );
            (v: $path:expr, $ext:expr, $exp:expr) => (
                {
                    let mut path = Path::from_vec($path);
                    path.add_extension($ext);
                    assert_eq!(path.as_vec(), $exp);
                }
            )
        )

        t!(s: "hi/there.txt", "foo", "hi/there.txt.foo");
        t!(s: "hi/there.", "foo", "hi/there..foo");
        t!(s: "hi/there", ".foo", "hi/there..foo");
        t!(s: "hi/there.txt", "", "hi/there.txt");
        t!(v: b!("hi/there.txt"), b!("foo"), b!("hi/there.txt.foo"));
        t!(v: b!("hi/there"), b!("bar"), b!("hi/there.bar"));
        t!(v: b!("/"), b!("foo"), b!("/"));
        t!(v: b!("."), b!("foo"), b!("."));
        t!(v: b!("hi/there.", 0x80, "foo"), b!(0xff), b!("hi/there.", 0x80, "foo.", 0xff));
    }

    #[test]
    fn test_getters() {
        macro_rules! t(
            (s: $path:expr, $filename:expr, $dirname:expr, $filestem:expr, $ext:expr) => (
                {
                    let path = $path;
                    let filename = $filename;
                    assert!(path.filename_str() == filename,
                            "`%s`.filename_str(): Expected `%?`, found `%?`",
                            path.as_str().unwrap(), filename, path.filename_str());
                    let dirname = $dirname;
                    assert!(path.dirname_str() == dirname,
                            "`%s`.dirname_str(): Expected `%?`, found `%?`",
                            path.as_str().unwrap(), dirname, path.dirname_str());
                    let filestem = $filestem;
                    assert!(path.filestem_str() == filestem,
                            "`%s`.filestem_str(): Expected `%?`, found `%?`",
                            path.as_str().unwrap(), filestem, path.filestem_str());
                    let ext = $ext;
                    assert!(path.extension_str() == ext,
                            "`%s`.extension_str(): Expected `%?`, found `%?`",
                            path.as_str().unwrap(), ext, path.extension_str());
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

        t!(v: Path::from_vec(b!("a/b/c")), Some(b!("c")), b!("a/b"), Some(b!("c")), None);
        t!(v: Path::from_vec(b!("a/b/", 0xff)), Some(b!(0xff)), b!("a/b"), Some(b!(0xff)), None);
        t!(v: Path::from_vec(b!("hi/there.", 0xff)), Some(b!("there.", 0xff)), b!("hi"),
              Some(b!("there")), Some(b!(0xff)));
        t!(s: Path::from_str("a/b/c"), Some("c"), Some("a/b"), Some("c"), None);
        t!(s: Path::from_str("."), None, Some("."), None, None);
        t!(s: Path::from_str("/"), None, Some("/"), None, None);
        t!(s: Path::from_str(".."), None, Some(".."), None, None);
        t!(s: Path::from_str("../.."), None, Some("../.."), None, None);
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
        macro_rules! t(
            (s: $path:expr, $abs:expr, $rel:expr) => (
                {
                    let path = Path::from_str($path);
                    assert_eq!(path.is_absolute(), $abs);
                    assert_eq!(path.is_relative(), $rel);
                }
            )
        )
        t!(s: "a/b/c", false, true);
        t!(s: "/a/b/c", true, false);
        t!(s: "a", false, true);
        t!(s: "/a", true, false);
        t!(s: ".", false, true);
        t!(s: "/", true, false);
        t!(s: "..", false, true);
        t!(s: "../..", false, true);
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
    fn test_ends_with_path() {
        macro_rules! t(
            (s: $path:expr, $child:expr, $exp:expr) => (
                {
                    let path = Path::from_str($path);
                    let child = Path::from_str($child);
                    assert_eq!(path.ends_with_path(&child), $exp);
                }
            );
            (v: $path:expr, $child:expr, $exp:expr) => (
                {
                    let path = Path::from_vec($path);
                    let child = Path::from_vec($child);
                    assert_eq!(path.ends_with_path(&child), $exp);
                }
            )
        )

        t!(s: "a/b/c", "c", true);
        t!(s: "a/b/c", "d", false);
        t!(s: "foo/bar/quux", "bar", false);
        t!(s: "foo/bar/quux", "barquux", false);
        t!(s: "a/b/c", "b/c", true);
        t!(s: "a/b/c", "a/b/c", true);
        t!(s: "a/b/c", "foo/a/b/c", false);
        t!(s: "/a/b/c", "a/b/c", true);
        t!(s: "/a/b/c", "/a/b/c", false); // child must be relative
        t!(s: "/a/b/c", "foo/a/b/c", false);
        t!(s: "a/b/c", "", false);
        t!(s: "", "", true);
        t!(s: "/a/b/c", "d/e/f", false);
        t!(s: "a/b/c", "a/b", false);
        t!(s: "a/b/c", "b", false);
        t!(v: b!("a/b/c"), b!("b/c"), true);
        t!(v: b!("a/b/", 0xff), b!(0xff), true);
        t!(v: b!("a/b/", 0xff), b!("b/", 0xff), true);
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
                    assert!(comps == exps, "component_iter: Expected %?, found %?",
                            comps, exps);
                    let comps = path.rev_component_iter().to_owned_vec();
                    let exps = exps.move_rev_iter().to_owned_vec();
                    assert!(comps == exps, "rev_component_iter: Expected %?, found %?",
                            comps, exps);
                }
            );
            (v: [$($arg:expr),+], [$([$($exp:expr),*]),*]) => (
                {
                    let path = Path::from_vec(b!($($arg),+));
                    let comps = path.component_iter().to_owned_vec();
                    let exp: &[&[u8]] = [$(b!($($exp),*)),*];
                    assert!(comps.as_slice() == exp, "component_iter: Expected %?, found %?",
                            comps.as_slice(), exp);
                    let comps = path.rev_component_iter().to_owned_vec();
                    let exp = exp.rev_iter().map(|&x|x).to_owned_vec();
                    assert!(comps.as_slice() == exp, "rev_component_iter: Expected %?, found %?",
                            comps.as_slice(), exp);
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

    #[test]
    fn test_str_component_iter() {
        macro_rules! t(
            (v: [$($arg:expr),+], $exp:expr) => (
                {
                    let path = Path::from_vec(b!($($arg),+));
                    let comps = path.str_component_iter().to_owned_vec();
                    let exp: &[Option<&str>] = $exp;
                    assert!(comps.as_slice() == exp, "str_component_iter: Expected %?, found %?",
                            comps.as_slice(), exp);
                    let comps = path.rev_str_component_iter().to_owned_vec();
                    let exp = exp.rev_iter().map(|&x|x).to_owned_vec();
                    assert!(comps.as_slice() == exp,
                            "rev_str_component_iter: Expected %?, found %?",
                            comps.as_slice(), exp);
                }
            )
        )

        t!(v: ["a/b/c"], [Some("a"), Some("b"), Some("c")]);
        t!(v: ["/", 0xff, "/a/", 0x80], [None, Some("a"), None]);
        t!(v: ["../../foo", 0xcd, "bar"], [Some(".."), Some(".."), None]);
        // str_component_iter is a wrapper around component_iter, so no need to do
        // the full set of tests
    }

    #[test]
    fn test_each_parent() {
        assert!(Path::from_str("/foo/bar").each_parent(|_| true));
        assert!(!Path::from_str("/foo/bar").each_parent(|_| false));

        macro_rules! t(
            (s: $path:expr, $exp:expr) => (
                {
                    let path = Path::from_str($path);
                    let exp: &[&str] = $exp;
                    let mut comps = exp.iter().map(|&x|x);
                    do path.each_parent |p| {
                        let p = p.as_str();
                        assert!(p.is_some());
                        let e = comps.next();
                        assert!(e.is_some());
                        assert_eq!(p.unwrap(), e.unwrap());
                        true
                    };
                    assert!(comps.next().is_none());
                }
            );
            (v: $path:expr, $exp:expr) => (
                {
                    let path = Path::from_vec($path);
                    let exp: &[&[u8]] = $exp;
                    let mut comps = exp.iter().map(|&x|x);
                    do path.each_parent |p| {
                        let p = p.as_vec();
                        let e = comps.next();
                        assert!(e.is_some());
                        assert_eq!(p, e.unwrap());
                        true
                    };
                    assert!(comps.next().is_none());
                }
            )
        )

        t!(s: "/foo/bar", ["/foo/bar", "/foo", "/"]);
        t!(s: "/foo/bar/baz", ["/foo/bar/baz", "/foo/bar", "/foo", "/"]);
        t!(s: "/foo", ["/foo", "/"]);
        t!(s: "/", ["/"]);
        t!(s: "foo/bar/baz", ["foo/bar/baz", "foo/bar", "foo", "."]);
        t!(s: "foo/bar", ["foo/bar", "foo", "."]);
        t!(s: "foo", ["foo", "."]);
        t!(s: ".", ["."]);
        t!(s: "..", [".."]);
        t!(s: "../../foo", ["../../foo", "../.."]);

        t!(v: b!("foo/bar", 0x80), [b!("foo/bar", 0x80), b!("foo"), b!(".")]);
        t!(v: b!(0xff, "/bar"), [b!(0xff, "/bar"), b!(0xff), b!(".")]);
    }
}
