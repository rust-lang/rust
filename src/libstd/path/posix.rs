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
use vec;
use vec::{CopyableVector, RSplitIterator, SplitIterator, Vector, VectorVector};
use super::{BytesContainer, GenericPath, GenericPathUnsafe};

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
pub static sep: char = '/';
static sep_byte: u8 = sep as u8;

/// Returns whether the given byte is a path separator
#[inline]
pub fn is_sep_byte(u: &u8) -> bool {
    *u as char == sep
}

/// Returns whether the given char is a path separator
#[inline]
pub fn is_sep(c: char) -> bool {
    c == sep
}

impl Eq for Path {
    #[inline]
    fn eq(&self, other: &Path) -> bool {
        self.repr == other.repr
    }
}

impl FromStr for Path {
    fn from_str(s: &str) -> Option<Path> {
        Path::init_opt(s)
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
    fn iter_bytes(&self, lsb0: bool, f: |buf: &[u8]| -> bool) -> bool {
        self.repr.iter_bytes(lsb0, f)
    }
}

impl BytesContainer for Path {
    #[inline]
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] {
        self.as_vec()
    }
    #[inline]
    fn container_into_owned_bytes(self) -> ~[u8] {
        self.into_vec()
    }
}

impl<'self> BytesContainer for &'self Path {
    #[inline]
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] {
        self.as_vec()
    }
}

impl GenericPathUnsafe for Path {
    unsafe fn init_unchecked<T: BytesContainer>(path: T) -> Path {
        let path = Path::normalize(path.container_as_bytes());
        assert!(!path.is_empty());
        let idx = path.rposition_elem(&sep_byte);
        Path{ repr: path, sepidx: idx }
    }

    unsafe fn set_filename_unchecked<T: BytesContainer>(&mut self, filename: T) {
        let filename = filename.container_as_bytes();
        match self.sepidx {
            None if bytes!("..") == self.repr => {
                let mut v = vec::with_capacity(3 + filename.len());
                v.push_all(dot_dot_static);
                v.push(sep_byte);
                v.push_all(filename);
                self.repr = Path::normalize(v);
            }
            None => {
                self.repr = Path::normalize(filename);
            }
            Some(idx) if self.repr.slice_from(idx+1) == bytes!("..") => {
                let mut v = vec::with_capacity(self.repr.len() + 1 + filename.len());
                v.push_all(self.repr);
                v.push(sep_byte);
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
        self.sepidx = self.repr.rposition_elem(&sep_byte);
    }

    unsafe fn push_unchecked<T: BytesContainer>(&mut self, path: T) {
        let path = path.container_as_bytes();
        if !path.is_empty() {
            if path[0] == sep_byte {
                self.repr = Path::normalize(path);
            }  else {
                let mut v = vec::with_capacity(self.repr.len() + path.len() + 1);
                v.push_all(self.repr);
                v.push(sep_byte);
                v.push_all(path);
                self.repr = Path::normalize(v);
            }
            self.sepidx = self.repr.rposition_elem(&sep_byte);
        }
    }
}

impl GenericPath for Path {
    #[inline]
    fn as_vec<'a>(&'a self) -> &'a [u8] {
        self.repr.as_slice()
    }

    fn into_vec(self) -> ~[u8] {
        self.repr
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

    fn pop(&mut self) -> bool {
        match self.sepidx {
            None if bytes!(".") == self.repr => false,
            None => {
                self.repr = ~['.' as u8];
                self.sepidx = None;
                true
            }
            Some(0) if bytes!("/") == self.repr => false,
            Some(idx) => {
                if idx == 0 {
                    self.repr.truncate(idx+1);
                } else {
                    self.repr.truncate(idx);
                }
                self.sepidx = self.repr.rposition_elem(&sep_byte);
                true
            }
        }
    }

    fn root_path(&self) -> Option<Path> {
        if self.is_absolute() {
            Some(Path::init("/"))
        } else {
            None
        }
    }

    #[inline]
    fn is_absolute(&self) -> bool {
        self.repr[0] == sep_byte
    }

    fn is_ancestor_of(&self, other: &Path) -> bool {
        if self.is_absolute() != other.is_absolute() {
            false
        } else {
            let mut ita = self.components();
            let mut itb = other.components();
            if bytes!(".") == self.repr {
                return itb.next() != Some(bytes!(".."));
            }
            loop {
                match (ita.next(), itb.next()) {
                    (None, _) => break,
                    (Some(a), Some(b)) if a == b => { continue },
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
            let mut ita = self.components();
            let mut itb = base.components();
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
            Some(Path::init(comps.connect_vec(&sep_byte)))
        }
    }

    fn ends_with_path(&self, child: &Path) -> bool {
        if !child.is_relative() { return false; }
        let mut selfit = self.rev_components();
        let mut childit = child.rev_components();
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

impl Path {
    /// Returns a new Path from a byte vector or string
    ///
    /// # Failure
    ///
    /// Raises the `null_byte` condition if the vector contains a NUL.
    #[inline]
    pub fn init<T: BytesContainer>(path: T) -> Path {
        GenericPath::init(path)
    }

    /// Returns a new Path from a byte vector or string, if possible
    #[inline]
    pub fn init_opt<T: BytesContainer>(path: T) -> Option<Path> {
        GenericPath::init_opt(path)
    }

    /// Returns a normalized byte vector representation of a path, by removing all empty
    /// components, and unnecessary . and .. components.
    fn normalize<V: Vector<u8>+CopyableVector<u8>>(v: V) -> ~[u8] {
        // borrowck is being very picky
        let val = {
            let is_abs = !v.as_slice().is_empty() && v.as_slice()[0] == sep_byte;
            let v_ = if is_abs { v.as_slice().slice_from(1) } else { v.as_slice() };
            let comps = normalize_helper(v_, is_abs);
            match comps {
                None => None,
                Some(comps) => {
                    if is_abs && comps.is_empty() {
                        Some(~[sep_byte])
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
                            v.push(sep_byte);
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
    pub fn components<'a>(&'a self) -> ComponentIter<'a> {
        let v = if self.repr[0] == sep_byte {
            self.repr.slice_from(1)
        } else { self.repr.as_slice() };
        let mut ret = v.split(is_sep_byte);
        if v.is_empty() {
            // consume the empty "" component
            ret.next();
        }
        ret
    }

    /// Returns an iterator that yields each component of the path in reverse.
    /// See components() for details.
    pub fn rev_components<'a>(&'a self) -> RevComponentIter<'a> {
        let v = if self.repr[0] == sep_byte {
            self.repr.slice_from(1)
        } else { self.repr.as_slice() };
        let mut ret = v.rsplit(is_sep_byte);
        if v.is_empty() {
            // consume the empty "" component
            ret.next();
        }
        ret
    }

    /// Returns an iterator that yields each component of the path as Option<&str>.
    /// See components() for details.
    pub fn str_components<'a>(&'a self) -> StrComponentIter<'a> {
        self.components().map(str::from_utf8_opt)
    }

    /// Returns an iterator that yields each component of the path in reverse as Option<&str>.
    /// See components() for details.
    pub fn rev_str_components<'a>(&'a self) -> RevStrComponentIter<'a> {
        self.rev_components().map(str::from_utf8_opt)
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
    for comp in v.split(is_sep_byte) {
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

static dot_static: &'static [u8] = bytes!(".");
static dot_dot_static: &'static [u8] = bytes!("..");

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
        let empty: &[u8] = [];
        t!(v: Path::init(empty), b!("."));
        t!(v: Path::init(b!("/")), b!("/"));
        t!(v: Path::init(b!("a/b/c")), b!("a/b/c"));
        t!(v: Path::init(b!("a/b/c", 0xff)), b!("a/b/c", 0xff));
        t!(v: Path::init(b!(0xff, "/../foo", 0x80)), b!("foo", 0x80));
        let p = Path::init(b!("a/b/c", 0xff));
        assert_eq!(p.as_str(), None);

        t!(s: Path::init(""), ".");
        t!(s: Path::init("/"), "/");
        t!(s: Path::init("hi"), "hi");
        t!(s: Path::init("hi/"), "hi");
        t!(s: Path::init("/lib"), "/lib");
        t!(s: Path::init("/lib/"), "/lib");
        t!(s: Path::init("hi/there"), "hi/there");
        t!(s: Path::init("hi/there.txt"), "hi/there.txt");

        t!(s: Path::init("hi/there/"), "hi/there");
        t!(s: Path::init("hi/../there"), "there");
        t!(s: Path::init("../hi/there"), "../hi/there");
        t!(s: Path::init("/../hi/there"), "/hi/there");
        t!(s: Path::init("foo/.."), ".");
        t!(s: Path::init("/foo/.."), "/");
        t!(s: Path::init("/foo/../.."), "/");
        t!(s: Path::init("/foo/../../bar"), "/bar");
        t!(s: Path::init("/./hi/./there/."), "/hi/there");
        t!(s: Path::init("/./hi/./there/./.."), "/hi");
        t!(s: Path::init("foo/../.."), "..");
        t!(s: Path::init("foo/../../.."), "../..");
        t!(s: Path::init("foo/../../bar"), "../bar");

        assert_eq!(Path::init(b!("foo/bar")).into_vec(), b!("foo/bar").to_owned());
        assert_eq!(Path::init(b!("/foo/../../bar")).into_vec(),
                   b!("/bar").to_owned());

        let p = Path::init(b!("foo/bar", 0x80));
        assert_eq!(p.as_str(), None);
    }

    #[test]
    fn test_opt_paths() {
        assert_eq!(Path::init_opt(b!("foo/bar", 0)), None);
        t!(v: Path::init_opt(b!("foo/bar")).unwrap(), b!("foo/bar"));
        assert_eq!(Path::init_opt("foo/bar\0"), None);
        t!(s: Path::init_opt("foo/bar").unwrap(), "foo/bar");
    }

    #[test]
    fn test_null_byte() {
        use path::null_byte::cond;

        let mut handled = false;
        let mut p = cond.trap(|v| {
            handled = true;
            assert_eq!(v.as_slice(), b!("foo/bar", 0));
            (b!("/bar").to_owned())
        }).inside(|| {
            Path::init(b!("foo/bar", 0))
        });
        assert!(handled);
        assert_eq!(p.as_vec(), b!("/bar"));

        handled = false;
        cond.trap(|v| {
            handled = true;
            assert_eq!(v.as_slice(), b!("f", 0, "o"));
            (b!("foo").to_owned())
        }).inside(|| {
            p.set_filename(b!("f", 0, "o"))
        });
        assert!(handled);
        assert_eq!(p.as_vec(), b!("/foo"));

        handled = false;
        cond.trap(|v| {
            handled = true;
            assert_eq!(v.as_slice(), b!("f", 0, "o"));
            (b!("foo").to_owned())
        }).inside(|| {
            p.push(b!("f", 0, "o"));
        });
        assert!(handled);
        assert_eq!(p.as_vec(), b!("/foo/foo"));
    }

    #[test]
    fn test_null_byte_fail() {
        use path::null_byte::cond;
        use task;

        macro_rules! t(
            ($name:expr => $code:block) => (
                {
                    let mut t = task::task();
                    t.name($name);
                    let res = do t.try $code;
                    assert!(res.is_err());
                }
            )
        )

        t!(~"new() w/nul" => {
            cond.trap(|_| {
                (b!("null", 0).to_owned())
            }).inside(|| {
                Path::init(b!("foo/bar", 0))
            });
        })

        t!(~"set_filename w/nul" => {
            let mut p = Path::init(b!("foo/bar"));
            cond.trap(|_| {
                (b!("null", 0).to_owned())
            }).inside(|| {
                p.set_filename(b!("foo", 0))
            });
        })

        t!(~"push w/nul" => {
            let mut p = Path::init(b!("foo/bar"));
            cond.trap(|_| {
                (b!("null", 0).to_owned())
            }).inside(|| {
                p.push(b!("foo", 0))
            });
        })
    }

    #[test]
    fn test_display_str() {
        macro_rules! t(
            ($path:expr, $disp:ident, $exp:expr) => (
                {
                    let path = Path::init($path);
                    assert_eq!(path.$disp().to_str(), ~$exp);
                }
            )
        )
        t!("foo", display, "foo");
        t!(b!("foo", 0x80), display, "foo\uFFFD");
        t!(b!("foo", 0xff, "bar"), display, "foo\uFFFDbar");
        t!(b!("foo", 0xff, "/bar"), filename_display, "bar");
        t!(b!("foo/", 0xff, "bar"), filename_display, "\uFFFDbar");
        t!(b!("/"), filename_display, "");

        macro_rules! t(
            ($path:expr, $exp:expr) => (
                {
                    let mut called = false;
                    let path = Path::init($path);
                    path.display().with_str(|s| {
                        assert_eq!(s, $exp);
                        called = true;
                    });
                    assert!(called);
                }
            );
            ($path:expr, $exp:expr, filename) => (
                {
                    let mut called = false;
                    let path = Path::init($path);
                    path.filename_display().with_str(|s| {
                        assert_eq!(s, $exp);
                        called = true;
                    });
                    assert!(called);
                }
            )
        )

        t!("foo", "foo");
        t!(b!("foo", 0x80), "foo\uFFFD");
        t!(b!("foo", 0xff, "bar"), "foo\uFFFDbar");
        t!(b!("foo", 0xff, "/bar"), "bar", filename);
        t!(b!("foo/", 0xff, "bar"), "\uFFFDbar", filename);
        t!(b!("/"), "", filename);
    }

    #[test]
    fn test_display() {
        macro_rules! t(
            ($path:expr, $exp:expr, $expf:expr) => (
                {
                    let path = Path::init($path);
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
                    let path = Path::init($path);
                    assert_eq!(path.$op(), ($exp).as_bytes());
                }
            );
            (s: $path:expr, $op:ident, $exp:expr, opt) => (
                {
                    let path = Path::init($path);
                    let left = path.$op().map(|x| str::from_utf8(x));
                    assert_eq!(left, $exp);
                }
            );
            (v: $path:expr, $op:ident, $exp:expr) => (
                {
                    let path = Path::init($path);
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
                    let mut p1 = Path::init(path);
                    let p2 = p1.clone();
                    p1.push(join);
                    assert_eq!(p1, p2.join(join));
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
                    let mut p = Path::init($path);
                    let push = Path::init($push);
                    p.push(&push);
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
                    let mut p = Path::init($path);
                    p.push_many($push);
                    assert_eq!(p.as_str(), Some($exp));
                }
            );
            (v: $path:expr, $push:expr, $exp:expr) => (
                {
                    let mut p = Path::init($path);
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
                    let mut p = Path::init($path);
                    let result = p.pop();
                    assert_eq!(p.as_str(), Some($left));
                    assert_eq!(result, $right);
                }
            );
            (v: [$($path:expr),+], [$($left:expr),+], $right:expr) => (
                {
                    let mut p = Path::init(b!($($path),+));
                    let result = p.pop();
                    assert_eq!(p.as_vec(), b!($($left),+));
                    assert_eq!(result, $right);
                }
            )
        )

        t!(v: ["a/b/c"], ["a/b"], true);
        t!(v: ["a"], ["."], true);
        t!(v: ["."], ["."], false);
        t!(v: ["/a"], ["/"], true);
        t!(v: ["/"], ["/"], false);
        t!(v: ["a/b/c", 0x80], ["a/b"], true);
        t!(v: ["a/b", 0x80, "/c"], ["a/b", 0x80], true);
        t!(v: [0xff], ["."], true);
        t!(v: ["/", 0xff], ["/"], true);
        t!(s: "a/b/c", "a/b", true);
        t!(s: "a", ".", true);
        t!(s: ".", ".", false);
        t!(s: "/a", "/", true);
        t!(s: "/", "/", false);
    }

    #[test]
    fn test_root_path() {
        assert_eq!(Path::init(b!("a/b/c")).root_path(), None);
        assert_eq!(Path::init(b!("/a/b/c")).root_path(), Some(Path::init("/")));
    }

    #[test]
    fn test_join() {
        t!(v: Path::init(b!("a/b/c")).join(b!("..")), b!("a/b"));
        t!(v: Path::init(b!("/a/b/c")).join(b!("d")), b!("/a/b/c/d"));
        t!(v: Path::init(b!("a/", 0x80, "/c")).join(b!(0xff)), b!("a/", 0x80, "/c/", 0xff));
        t!(s: Path::init("a/b/c").join(".."), "a/b");
        t!(s: Path::init("/a/b/c").join("d"), "/a/b/c/d");
        t!(s: Path::init("a/b").join("c/d"), "a/b/c/d");
        t!(s: Path::init("a/b").join("/c/d"), "/c/d");
        t!(s: Path::init(".").join("a/b"), "a/b");
        t!(s: Path::init("/").join("a/b"), "/a/b");
    }

    #[test]
    fn test_join_path() {
        macro_rules! t(
            (s: $path:expr, $join:expr, $exp:expr) => (
                {
                    let path = Path::init($path);
                    let join = Path::init($join);
                    let res = path.join(&join);
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
                    let path = Path::init($path);
                    let res = path.join_many($join);
                    assert_eq!(res.as_str(), Some($exp));
                }
            );
            (v: $path:expr, $join:expr, $exp:expr) => (
                {
                    let path = Path::init($path);
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
        let empty: &[u8] = [];

        t!(v: Path::init(b!("a/b/c")).with_filename(b!("d")), b!("a/b/d"));
        t!(v: Path::init(b!("a/b/c", 0xff)).with_filename(b!(0x80)), b!("a/b/", 0x80));
        t!(v: Path::init(b!("/", 0xff, "/foo")).with_filename(b!(0xcd)),
              b!("/", 0xff, "/", 0xcd));
        t!(s: Path::init("a/b/c").with_filename("d"), "a/b/d");
        t!(s: Path::init(".").with_filename("foo"), "foo");
        t!(s: Path::init("/a/b/c").with_filename("d"), "/a/b/d");
        t!(s: Path::init("/").with_filename("foo"), "/foo");
        t!(s: Path::init("/a").with_filename("foo"), "/foo");
        t!(s: Path::init("foo").with_filename("bar"), "bar");
        t!(s: Path::init("/").with_filename("foo/"), "/foo");
        t!(s: Path::init("/a").with_filename("foo/"), "/foo");
        t!(s: Path::init("a/b/c").with_filename(""), "a/b");
        t!(s: Path::init("a/b/c").with_filename("."), "a/b");
        t!(s: Path::init("a/b/c").with_filename(".."), "a");
        t!(s: Path::init("/a").with_filename(""), "/");
        t!(s: Path::init("foo").with_filename(""), ".");
        t!(s: Path::init("a/b/c").with_filename("d/e"), "a/b/d/e");
        t!(s: Path::init("a/b/c").with_filename("/d"), "a/b/d");
        t!(s: Path::init("..").with_filename("foo"), "../foo");
        t!(s: Path::init("../..").with_filename("foo"), "../../foo");
        t!(s: Path::init("..").with_filename(""), "..");
        t!(s: Path::init("../..").with_filename(""), "../..");

        t!(v: Path::init(b!("hi/there", 0x80, ".txt")).with_extension(b!("exe")),
              b!("hi/there", 0x80, ".exe"));
        t!(v: Path::init(b!("hi/there.txt", 0x80)).with_extension(b!(0xff)),
              b!("hi/there.", 0xff));
        t!(v: Path::init(b!("hi/there", 0x80)).with_extension(b!(0xff)),
              b!("hi/there", 0x80, ".", 0xff));
        t!(v: Path::init(b!("hi/there.", 0xff)).with_extension(empty), b!("hi/there"));
        t!(s: Path::init("hi/there.txt").with_extension("exe"), "hi/there.exe");
        t!(s: Path::init("hi/there.txt").with_extension(""), "hi/there");
        t!(s: Path::init("hi/there.txt").with_extension("."), "hi/there..");
        t!(s: Path::init("hi/there.txt").with_extension(".."), "hi/there...");
        t!(s: Path::init("hi/there").with_extension("txt"), "hi/there.txt");
        t!(s: Path::init("hi/there").with_extension("."), "hi/there..");
        t!(s: Path::init("hi/there").with_extension(".."), "hi/there...");
        t!(s: Path::init("hi/there.").with_extension("txt"), "hi/there.txt");
        t!(s: Path::init("hi/.foo").with_extension("txt"), "hi/.foo.txt");
        t!(s: Path::init("hi/there.txt").with_extension(".foo"), "hi/there..foo");
        t!(s: Path::init("/").with_extension("txt"), "/");
        t!(s: Path::init("/").with_extension("."), "/");
        t!(s: Path::init("/").with_extension(".."), "/");
        t!(s: Path::init(".").with_extension("txt"), ".");
    }

    #[test]
    fn test_setters() {
        macro_rules! t(
            (s: $path:expr, $set:ident, $with:ident, $arg:expr) => (
                {
                    let path = $path;
                    let arg = $arg;
                    let mut p1 = Path::init(path);
                    p1.$set(arg);
                    let p2 = Path::init(path);
                    assert_eq!(p1, p2.$with(arg));
                }
            );
            (v: $path:expr, $set:ident, $with:ident, $arg:expr) => (
                {
                    let path = $path;
                    let arg = $arg;
                    let mut p1 = Path::init(path);
                    p1.$set(arg);
                    let p2 = Path::init(path);
                    assert_eq!(p1, p2.$with(arg));
                }
            )
        )

        t!(v: b!("a/b/c"), set_filename, with_filename, b!("d"));
        t!(v: b!("/"), set_filename, with_filename, b!("foo"));
        t!(v: b!(0x80), set_filename, with_filename, b!(0xff));
        t!(s: "a/b/c", set_filename, with_filename, "d");
        t!(s: "/", set_filename, with_filename, "foo");
        t!(s: ".", set_filename, with_filename, "foo");
        t!(s: "a/b", set_filename, with_filename, "");
        t!(s: "a", set_filename, with_filename, "");

        t!(v: b!("hi/there.txt"), set_extension, with_extension, b!("exe"));
        t!(v: b!("hi/there.t", 0x80, "xt"), set_extension, with_extension, b!("exe", 0xff));
        t!(s: "hi/there.txt", set_extension, with_extension, "exe");
        t!(s: "hi/there.", set_extension, with_extension, "txt");
        t!(s: "hi/there", set_extension, with_extension, "txt");
        t!(s: "hi/there.txt", set_extension, with_extension, "");
        t!(s: "hi/there", set_extension, with_extension, "");
        t!(s: ".", set_extension, with_extension, "txt");
    }

    #[test]
    fn test_getters() {
        macro_rules! t(
            (s: $path:expr, $filename:expr, $dirname:expr, $filestem:expr, $ext:expr) => (
                {
                    let path = $path;
                    let filename = $filename;
                    assert!(path.filename_str() == filename,
                            "{}.filename_str(): Expected `{:?}`, found {:?}",
                            path.as_str().unwrap(), filename, path.filename_str());
                    let dirname = $dirname;
                    assert!(path.dirname_str() == dirname,
                            "`{}`.dirname_str(): Expected `{:?}`, found `{:?}`",
                            path.as_str().unwrap(), dirname, path.dirname_str());
                    let filestem = $filestem;
                    assert!(path.filestem_str() == filestem,
                            "`{}`.filestem_str(): Expected `{:?}`, found `{:?}`",
                            path.as_str().unwrap(), filestem, path.filestem_str());
                    let ext = $ext;
                    assert!(path.extension_str() == ext,
                            "`{}`.extension_str(): Expected `{:?}`, found `{:?}`",
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

        t!(v: Path::init(b!("a/b/c")), Some(b!("c")), b!("a/b"), Some(b!("c")), None);
        t!(v: Path::init(b!("a/b/", 0xff)), Some(b!(0xff)), b!("a/b"), Some(b!(0xff)), None);
        t!(v: Path::init(b!("hi/there.", 0xff)), Some(b!("there.", 0xff)), b!("hi"),
              Some(b!("there")), Some(b!(0xff)));
        t!(s: Path::init("a/b/c"), Some("c"), Some("a/b"), Some("c"), None);
        t!(s: Path::init("."), None, Some("."), None, None);
        t!(s: Path::init("/"), None, Some("/"), None, None);
        t!(s: Path::init(".."), None, Some(".."), None, None);
        t!(s: Path::init("../.."), None, Some("../.."), None, None);
        t!(s: Path::init("hi/there.txt"), Some("there.txt"), Some("hi"),
              Some("there"), Some("txt"));
        t!(s: Path::init("hi/there"), Some("there"), Some("hi"), Some("there"), None);
        t!(s: Path::init("hi/there."), Some("there."), Some("hi"),
              Some("there"), Some(""));
        t!(s: Path::init("hi/.there"), Some(".there"), Some("hi"), Some(".there"), None);
        t!(s: Path::init("hi/..there"), Some("..there"), Some("hi"),
              Some("."), Some("there"));
        t!(s: Path::init(b!("a/b/", 0xff)), None, Some("a/b"), None, None);
        t!(s: Path::init(b!("a/b/", 0xff, ".txt")), None, Some("a/b"), None, Some("txt"));
        t!(s: Path::init(b!("a/b/c.", 0x80)), None, Some("a/b"), Some("c"), None);
        t!(s: Path::init(b!(0xff, "/b")), Some("b"), None, Some("b"), None);
    }

    #[test]
    fn test_dir_path() {
        t!(v: Path::init(b!("hi/there", 0x80)).dir_path(), b!("hi"));
        t!(v: Path::init(b!("hi", 0xff, "/there")).dir_path(), b!("hi", 0xff));
        t!(s: Path::init("hi/there").dir_path(), "hi");
        t!(s: Path::init("hi").dir_path(), ".");
        t!(s: Path::init("/hi").dir_path(), "/");
        t!(s: Path::init("/").dir_path(), "/");
        t!(s: Path::init("..").dir_path(), "..");
        t!(s: Path::init("../..").dir_path(), "../..");
    }

    #[test]
    fn test_is_absolute() {
        macro_rules! t(
            (s: $path:expr, $abs:expr, $rel:expr) => (
                {
                    let path = Path::init($path);
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
                    let path = Path::init($path);
                    let dest = Path::init($dest);
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
                    let path = Path::init($path);
                    let child = Path::init($child);
                    assert_eq!(path.ends_with_path(&child), $exp);
                }
            );
            (v: $path:expr, $child:expr, $exp:expr) => (
                {
                    let path = Path::init($path);
                    let child = Path::init($child);
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
                    let path = Path::init($path);
                    let other = Path::init($other);
                    let res = path.path_relative_from(&other);
                    assert_eq!(res.as_ref().and_then(|x| x.as_str()), $exp);
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
    fn test_components_iter() {
        macro_rules! t(
            (s: $path:expr, $exp:expr) => (
                {
                    let path = Path::init($path);
                    let comps = path.components().to_owned_vec();
                    let exp: &[&str] = $exp;
                    let exps = exp.iter().map(|x| x.as_bytes()).to_owned_vec();
                    assert!(comps == exps, "components: Expected {:?}, found {:?}",
                            comps, exps);
                    let comps = path.rev_components().to_owned_vec();
                    let exps = exps.move_rev_iter().to_owned_vec();
                    assert!(comps == exps, "rev_components: Expected {:?}, found {:?}",
                            comps, exps);
                }
            );
            (v: [$($arg:expr),+], [$([$($exp:expr),*]),*]) => (
                {
                    let path = Path::init(b!($($arg),+));
                    let comps = path.components().to_owned_vec();
                    let exp: &[&[u8]] = [$(b!($($exp),*)),*];
                    assert!(comps.as_slice() == exp, "components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                    let comps = path.rev_components().to_owned_vec();
                    let exp = exp.rev_iter().map(|&x|x).to_owned_vec();
                    assert!(comps.as_slice() == exp,
                            "rev_components: Expected {:?}, found {:?}",
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
    fn test_str_components() {
        macro_rules! t(
            (v: [$($arg:expr),+], $exp:expr) => (
                {
                    let path = Path::init(b!($($arg),+));
                    let comps = path.str_components().to_owned_vec();
                    let exp: &[Option<&str>] = $exp;
                    assert!(comps.as_slice() == exp,
                            "str_components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                    let comps = path.rev_str_components().to_owned_vec();
                    let exp = exp.rev_iter().map(|&x|x).to_owned_vec();
                    assert!(comps.as_slice() == exp,
                            "rev_str_components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                }
            )
        )

        t!(v: ["a/b/c"], [Some("a"), Some("b"), Some("c")]);
        t!(v: ["/", 0xff, "/a/", 0x80], [None, Some("a"), None]);
        t!(v: ["../../foo", 0xcd, "bar"], [Some(".."), Some(".."), None]);
        // str_components is a wrapper around components, so no need to do
        // the full set of tests
    }
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use super::*;

    #[bench]
    fn join_home_dir(bh: &mut BenchHarness) {
        let posix_path = Path::init("/");
        bh.iter(|| {
            posix_path.join("home");
        });
    }

    #[bench]
    fn join_abs_path_home_dir(bh: &mut BenchHarness) {
        let posix_path = Path::init("/");
        bh.iter(|| {
            posix_path.join("/home");
        });
    }

    #[bench]
    fn join_many_home_dir(bh: &mut BenchHarness) {
        let posix_path = Path::init("/");
        bh.iter(|| {
            posix_path.join_many(&["home"]);
        });
    }

    #[bench]
    fn join_many_abs_path_home_dir(bh: &mut BenchHarness) {
        let posix_path = Path::init("/");
        bh.iter(|| {
            posix_path.join_many(&["/home"]);
        });
    }

    #[bench]
    fn push_home_dir(bh: &mut BenchHarness) {
        let mut posix_path = Path::init("/");
        bh.iter(|| {
            posix_path.push("home");
        });
    }

    #[bench]
    fn push_abs_path_home_dir(bh: &mut BenchHarness) {
        let mut posix_path = Path::init("/");
        bh.iter(|| {
            posix_path.push("/home");
        });
    }

    #[bench]
    fn push_many_home_dir(bh: &mut BenchHarness) {
        let mut posix_path = Path::init("/");
        bh.iter(|| {
            posix_path.push_many(&["home"]);
        });
    }

    #[bench]
    fn push_many_abs_path_home_dir(bh: &mut BenchHarness) {
        let mut posix_path = Path::init("/");
        bh.iter(|| {
            posix_path.push_many(&["/home"]);
        });
    }

    #[bench]
    fn ends_with_path_home_dir(bh: &mut BenchHarness) {
        let posix_home_path = Path::init("/home");
        bh.iter(|| {
            posix_home_path.ends_with_path(&Path::init("home"));
        });
    }

    #[bench]
    fn ends_with_path_missmatch_jome_home(bh: &mut BenchHarness) {
        let posix_home_path = Path::init("/home");
        bh.iter(|| {
            posix_home_path.ends_with_path(&Path::init("jome"));
        });
    }
}
