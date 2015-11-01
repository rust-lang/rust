// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Cross-platform path manipulation.
//!
//! This module provides two types, `PathBuf` and `Path` (akin to `String` and
//! `str`), for working with paths abstractly. These types are thin wrappers
//! around `OsString` and `OsStr` respectively, meaning that they work directly
//! on strings according to the local platform's path syntax.
//!
//! ## Simple usage
//!
//! Path manipulation includes both parsing components from slices and building
//! new owned paths.
//!
//! To parse a path, you can create a `Path` slice from a `str`
//! slice and start asking questions:
//!
//! ```rust
//! use std::path::Path;
//!
//! let path = Path::new("/tmp/foo/bar.txt");
//! let file = path.file_name();
//! let extension = path.extension();
//! let parent_dir = path.parent();
//! ```
//!
//! To build or modify paths, use `PathBuf`:
//!
//! ```rust
//! use std::path::PathBuf;
//!
//! let mut path = PathBuf::from("c:\\");
//! path.push("windows");
//! path.push("system32");
//! path.set_extension("dll");
//! ```
//!
//! ## Path components and normalization
//!
//! The path APIs are built around the notion of "components", which roughly
//! correspond to the substrings between path separators (`/` and, on Windows,
//! `\`). The APIs for path parsing are largely specified in terms of the path's
//! components, so it's important to clearly understand how those are determined.
//!
//! A path can always be reconstructed into an *equivalent* path by
//! putting together its components via `push`. Syntactically, the
//! paths may differ by the normalization described below.
//!
//! ### Component types
//!
//! Components come in several types:
//!
//! * Normal components are the default: standard references to files or
//! directories. The path `a/b` has two normal components, `a` and `b`.
//!
//! * Current directory components represent the `.` character. For example,
//! `./a` has a current directory component and a normal component `a`.
//!
//! * The root directory component represents a separator that designates
//!   starting from root. For example, `/a/b` has a root directory component
//!   followed by normal components `a` and `b`.
//!
//! On Windows, an additional component type comes into play:
//!
//! * Prefix components, of which there is a large variety. For example, `C:`
//! and `\\server\share` are prefixes. The path `C:windows` has a prefix
//! component `C:` and a normal component `windows`; the path `C:\windows` has a
//! prefix component `C:`, a root directory component, and a normal component
//! `windows`.
//!
//! ### Normalization
//!
//! Aside from splitting on the separator(s), there is a small amount of
//! "normalization":
//!
//! * Repeated separators are ignored: `a/b` and `a//b` both have components `a`
//!   and `b`.
//!
//! * Occurrences of `.` are normalized away, *except* if they are at
//! the beginning of the path (in which case they are often meaningful
//! in terms of path searching). So, for example, `a/./b`, `a/b/`,
//! `/a/b/.` and `a/b` all have components `a` and `b`, but `./a/b`
//! has a leading current directory component.
//!
//! No other normalization takes place by default. In particular,
//! `a/c` and `a/b/../c` are distinct, to account for the possibility
//! that `b` is a symbolic link (so its parent isn't `a`). Further
//! normalization is possible to build on top of the components APIs,
//! and will be included in this library in the near future.

#![stable(feature = "rust1", since = "1.0.0")]

use ascii::*;
use borrow::{Borrow, IntoCow, ToOwned, Cow};
use cmp;
use fmt;
use fs;
use io;
use iter;
use mem;
use ops::{self, Deref};
use string::String;
use vec::Vec;

use ffi::{OsStr, OsString};

use self::platform::{is_sep_byte, is_verbatim_sep, MAIN_SEP_STR, parse_prefix};

////////////////////////////////////////////////////////////////////////////////
// GENERAL NOTES
////////////////////////////////////////////////////////////////////////////////
//
// Parsing in this module is done by directly transmuting OsStr to [u8] slices,
// taking advantage of the fact that OsStr always encodes ASCII characters
// as-is.  Eventually, this transmutation should be replaced by direct uses of
// OsStr APIs for parsing, but it will take a while for those to become
// available.

////////////////////////////////////////////////////////////////////////////////
// Platform-specific definitions
////////////////////////////////////////////////////////////////////////////////

// The following modules give the most basic tools for parsing paths on various
// platforms. The bulk of the code is devoted to parsing prefixes on Windows.

#[cfg(unix)]
mod platform {
    use super::Prefix;
    use ffi::OsStr;

    #[inline]
    pub fn is_sep_byte(b: u8) -> bool {
        b == b'/'
    }

    #[inline]
    pub fn is_verbatim_sep(b: u8) -> bool {
        b == b'/'
    }

    pub fn parse_prefix(_: &OsStr) -> Option<Prefix> {
        None
    }

    pub const MAIN_SEP_STR: &'static str = "/";
    pub const MAIN_SEP: char = '/';
}

#[cfg(windows)]
mod platform {
    use ascii::*;

    use super::{os_str_as_u8_slice, u8_slice_as_os_str, Prefix};
    use ffi::OsStr;

    #[inline]
    pub fn is_sep_byte(b: u8) -> bool {
        b == b'/' || b == b'\\'
    }

    #[inline]
    pub fn is_verbatim_sep(b: u8) -> bool {
        b == b'\\'
    }

    pub fn parse_prefix<'a>(path: &'a OsStr) -> Option<Prefix> {
        use super::Prefix::*;
        unsafe {
            // The unsafety here stems from converting between &OsStr and &[u8]
            // and back. This is safe to do because (1) we only look at ASCII
            // contents of the encoding and (2) new &OsStr values are produced
            // only from ASCII-bounded slices of existing &OsStr values.
            let mut path = os_str_as_u8_slice(path);

            if path.starts_with(br"\\") {
                // \\
                path = &path[2..];
                if path.starts_with(br"?\") {
                    // \\?\
                    path = &path[2..];
                    if path.starts_with(br"UNC\") {
                        // \\?\UNC\server\share
                        path = &path[4..];
                        let (server, share) = match parse_two_comps(path, is_verbatim_sep) {
                            Some((server, share)) => (u8_slice_as_os_str(server),
                                                      u8_slice_as_os_str(share)),
                            None => (u8_slice_as_os_str(path),
                                     u8_slice_as_os_str(&[])),
                        };
                        return Some(VerbatimUNC(server, share));
                    } else {
                        // \\?\path
                        let idx = path.iter().position(|&b| b == b'\\');
                        if idx == Some(2) && path[1] == b':' {
                            let c = path[0];
                            if c.is_ascii() && (c as char).is_alphabetic() {
                                // \\?\C:\ path
                                return Some(VerbatimDisk(c.to_ascii_uppercase()));
                            }
                        }
                        let slice = &path[.. idx.unwrap_or(path.len())];
                        return Some(Verbatim(u8_slice_as_os_str(slice)));
                    }
                } else if path.starts_with(b".\\") {
                    // \\.\path
                    path = &path[2..];
                    let pos = path.iter().position(|&b| b == b'\\');
                    let slice = &path[..pos.unwrap_or(path.len())];
                    return Some(DeviceNS(u8_slice_as_os_str(slice)));
                }
                match parse_two_comps(path, is_sep_byte) {
                    Some((server, share)) if !server.is_empty() && !share.is_empty() => {
                        // \\server\share
                        return Some(UNC(u8_slice_as_os_str(server),
                                        u8_slice_as_os_str(share)));
                    }
                    _ => ()
                }
            } else if path.len() > 1 && path[1] == b':' {
                // C:
                let c = path[0];
                if c.is_ascii() && (c as char).is_alphabetic() {
                    return Some(Disk(c.to_ascii_uppercase()));
                }
            }
            return None;
        }

        fn parse_two_comps(mut path: &[u8], f: fn(u8) -> bool) -> Option<(&[u8], &[u8])> {
            let first = match path.iter().position(|x| f(*x)) {
                None => return None,
                Some(x) => &path[.. x]
            };
            path = &path[(first.len()+1)..];
            let idx = path.iter().position(|x| f(*x));
            let second = &path[.. idx.unwrap_or(path.len())];
            Some((first, second))
        }
    }

    pub const MAIN_SEP_STR: &'static str = "\\";
    pub const MAIN_SEP: char = '\\';
}

////////////////////////////////////////////////////////////////////////////////
// Windows Prefixes
////////////////////////////////////////////////////////////////////////////////

/// Path prefixes (Windows only).
///
/// Windows uses a variety of path styles, including references to drive
/// volumes (like `C:`), network shared folders (like `\\server\share`) and
/// others. In addition, some path prefixes are "verbatim", in which case
/// `/` is *not* treated as a separator and essentially no normalization is
/// performed.
#[derive(Copy, Clone, Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Prefix<'a> {
    /// Prefix `\\?\`, together with the given component immediately following it.
    #[stable(feature = "rust1", since = "1.0.0")]
    Verbatim(&'a OsStr),

    /// Prefix `\\?\UNC\`, with the "server" and "share" components following it.
    #[stable(feature = "rust1", since = "1.0.0")]
    VerbatimUNC(&'a OsStr, &'a OsStr),

    /// Prefix like `\\?\C:\`, for the given drive letter
    #[stable(feature = "rust1", since = "1.0.0")]
    VerbatimDisk(u8),

    /// Prefix `\\.\`, together with the given component immediately following it.
    #[stable(feature = "rust1", since = "1.0.0")]
    DeviceNS(&'a OsStr),

    /// Prefix `\\server\share`, with the given "server" and "share" components.
    #[stable(feature = "rust1", since = "1.0.0")]
    UNC(&'a OsStr, &'a OsStr),

    /// Prefix `C:` for the given disk drive.
    #[stable(feature = "rust1", since = "1.0.0")]
    Disk(u8),
}

impl<'a> Prefix<'a> {
    #[inline]
    fn len(&self) -> usize {
        use self::Prefix::*;
        fn os_str_len(s: &OsStr) -> usize {
            os_str_as_u8_slice(s).len()
        }
        match *self {
            Verbatim(x) => 4 + os_str_len(x),
            VerbatimUNC(x,y) => 8 + os_str_len(x) +
                if os_str_len(y) > 0 { 1 + os_str_len(y) }
                else { 0 },
            VerbatimDisk(_) => 6,
            UNC(x,y) => 2 + os_str_len(x) +
                if os_str_len(y) > 0 { 1 + os_str_len(y) }
                else { 0 },
            DeviceNS(x) => 4 + os_str_len(x),
            Disk(_) => 2
        }

    }

    /// Determines if the prefix is verbatim, i.e. begins with `\\?\`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_verbatim(&self) -> bool {
        use self::Prefix::*;
        match *self {
            Verbatim(_) | VerbatimDisk(_) | VerbatimUNC(_, _) => true,
            _ => false,
        }
    }

    #[inline]
    fn is_drive(&self) -> bool {
        match *self {
            Prefix::Disk(_) => true,
            _ => false,
        }
    }

    #[inline]
    fn has_implicit_root(&self) -> bool {
        !self.is_drive()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Exposed parsing helpers
////////////////////////////////////////////////////////////////////////////////

/// Determines whether the character is one of the permitted path
/// separators for the current platform.
///
/// # Examples
///
/// ```
/// use std::path;
///
/// assert!(path::is_separator('/'));
/// assert!(!path::is_separator('❤'));
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn is_separator(c: char) -> bool {
    c.is_ascii() && is_sep_byte(c as u8)
}

/// The primary separator for the current platform
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAIN_SEPARATOR: char = platform::MAIN_SEP;

////////////////////////////////////////////////////////////////////////////////
// Misc helpers
////////////////////////////////////////////////////////////////////////////////

// Iterate through `iter` while it matches `prefix`; return `None` if `prefix`
// is not a prefix of `iter`, otherwise return `Some(iter_after_prefix)` giving
// `iter` after having exhausted `prefix`.
fn iter_after<A, I, J>(mut iter: I, mut prefix: J) -> Option<I> where
    I: Iterator<Item=A> + Clone, J: Iterator<Item=A>, A: PartialEq
{
    loop {
        let mut iter_next = iter.clone();
        match (iter_next.next(), prefix.next()) {
            (Some(x), Some(y)) => {
                if x != y { return None }
            }
            (Some(_), None) => return Some(iter),
            (None, None) => return Some(iter),
            (None, Some(_)) => return None,
        }
        iter = iter_next;
    }
}

// See note at the top of this module to understand why these are used:
fn os_str_as_u8_slice(s: &OsStr) -> &[u8] {
    unsafe { mem::transmute(s) }
}
unsafe fn u8_slice_as_os_str(s: &[u8]) -> &OsStr {
    mem::transmute(s)
}

////////////////////////////////////////////////////////////////////////////////
// Cross-platform, iterator-independent parsing
////////////////////////////////////////////////////////////////////////////////

/// Says whether the first byte after the prefix is a separator.
fn has_physical_root(s: &[u8], prefix: Option<Prefix>) -> bool {
    let path = if let Some(p) = prefix { &s[p.len()..] } else { s };
    !path.is_empty() && is_sep_byte(path[0])
}

// basic workhorse for splitting stem and extension
fn split_file_at_dot(file: &OsStr) -> (Option<&OsStr>, Option<&OsStr>) {
    unsafe {
        if os_str_as_u8_slice(file) == b".." { return (Some(file), None) }

        // The unsafety here stems from converting between &OsStr and &[u8]
        // and back. This is safe to do because (1) we only look at ASCII
        // contents of the encoding and (2) new &OsStr values are produced
        // only from ASCII-bounded slices of existing &OsStr values.

        let mut iter = os_str_as_u8_slice(file).rsplitn(2, |b| *b == b'.');
        let after = iter.next();
        let before = iter.next();
        if before == Some(b"") {
            (Some(file), None)
        } else {
            (before.map(|s| u8_slice_as_os_str(s)),
             after.map(|s| u8_slice_as_os_str(s)))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// The core iterators
////////////////////////////////////////////////////////////////////////////////

/// Component parsing works by a double-ended state machine; the cursors at the
/// front and back of the path each keep track of what parts of the path have
/// been consumed so far.
///
/// Going front to back, a path is made up of a prefix, a starting
/// directory component, and a body (of normal components)
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
enum State {
    Prefix = 0,         // c:
    StartDir = 1,       // / or . or nothing
    Body = 2,           // foo/bar/baz
    Done = 3,
}

/// A Windows path prefix, e.g. `C:` or `\server\share`.
///
/// Does not occur on Unix.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone, Eq, Hash, Debug)]
pub struct PrefixComponent<'a> {
    /// The prefix as an unparsed `OsStr` slice.
    raw: &'a OsStr,

    /// The parsed prefix data.
    parsed: Prefix<'a>,
}

impl<'a> PrefixComponent<'a> {
    /// The parsed prefix data.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn kind(&self) -> Prefix<'a> {
        self.parsed
    }

    /// The raw `OsStr` slice for this prefix.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_os_str(&self) -> &'a OsStr {
        self.raw
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::PartialEq for PrefixComponent<'a> {
    fn eq(&self, other: &PrefixComponent<'a>) -> bool {
        cmp::PartialEq::eq(&self.parsed, &other.parsed)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::PartialOrd for PrefixComponent<'a> {
    fn partial_cmp(&self, other: &PrefixComponent<'a>) -> Option<cmp::Ordering> {
        cmp::PartialOrd::partial_cmp(&self.parsed, &other.parsed)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::Ord for PrefixComponent<'a> {
    fn cmp(&self, other: &PrefixComponent<'a>) -> cmp::Ordering {
        cmp::Ord::cmp(&self.parsed, &other.parsed)
    }
}

/// A single component of a path.
///
/// See the module documentation for an in-depth explanation of components and
/// their role in the API.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Component<'a> {
    /// A Windows path prefix, e.g. `C:` or `\server\share`.
    ///
    /// Does not occur on Unix.
    #[stable(feature = "rust1", since = "1.0.0")]
    Prefix(PrefixComponent<'a>),

    /// The root directory component, appears after any prefix and before anything else
    #[stable(feature = "rust1", since = "1.0.0")]
    RootDir,

    /// A reference to the current directory, i.e. `.`
    #[stable(feature = "rust1", since = "1.0.0")]
    CurDir,

    /// A reference to the parent directory, i.e. `..`
    #[stable(feature = "rust1", since = "1.0.0")]
    ParentDir,

    /// A normal component, i.e. `a` and `b` in `a/b`
    #[stable(feature = "rust1", since = "1.0.0")]
    Normal(&'a OsStr),
}

impl<'a> Component<'a> {
    /// Extracts the underlying `OsStr` slice
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_os_str(self) -> &'a OsStr {
        match self {
            Component::Prefix(p) => p.as_os_str(),
            Component::RootDir => OsStr::new(MAIN_SEP_STR),
            Component::CurDir => OsStr::new("."),
            Component::ParentDir => OsStr::new(".."),
            Component::Normal(path) => path,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> AsRef<OsStr> for Component<'a> {
    fn as_ref(&self) -> &OsStr {
        self.as_os_str()
    }
}

/// The core iterator giving the components of a path.
///
/// See the module documentation for an in-depth explanation of components and
/// their role in the API.
///
/// # Examples
///
/// ```
/// use std::path::Path;
///
/// let path = Path::new("/tmp/foo/bar.txt");
///
/// for component in path.components() {
///     println!("{:?}", component);
/// }
/// ```
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Components<'a> {
    // The path left to parse components from
    path: &'a [u8],

    // The prefix as it was originally parsed, if any
    prefix: Option<Prefix<'a>>,

    // true if path *physically* has a root separator; for most Windows
    // prefixes, it may have a "logical" rootseparator for the purposes of
    // normalization, e.g.  \\server\share == \\server\share\.
    has_physical_root: bool,

    // The iterator is double-ended, and these two states keep track of what has
    // been produced from either end
    front: State,
    back: State,
}

/// An iterator over the components of a path, as `OsStr` slices.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a> {
    inner: Components<'a>
}

impl<'a> Components<'a> {
    // how long is the prefix, if any?
    #[inline]
    fn prefix_len(&self) -> usize {
        self.prefix.as_ref().map(Prefix::len).unwrap_or(0)
    }

    #[inline]
    fn prefix_verbatim(&self) -> bool {
        self.prefix.as_ref().map(Prefix::is_verbatim).unwrap_or(false)
    }

    /// how much of the prefix is left from the point of view of iteration?
    #[inline]
    fn prefix_remaining(&self) -> usize {
        if self.front == State::Prefix {
            self.prefix_len()
        } else {
            0
        }
    }

    // Given the iteration so far, how much of the pre-State::Body path is left?
    #[inline]
    fn len_before_body(&self) -> usize {
        let root = if self.front <= State::StartDir && self.has_physical_root { 1 } else { 0 };
        let cur_dir = if self.front <= State::StartDir && self.include_cur_dir() { 1 } else { 0 };
        self.prefix_remaining() + root + cur_dir
    }

    // is the iteration complete?
    #[inline]
    fn finished(&self) -> bool {
        self.front == State::Done || self.back == State::Done || self.front > self.back
    }

    #[inline]
    fn is_sep_byte(&self, b: u8) -> bool {
        if self.prefix_verbatim() {
            is_verbatim_sep(b)
        } else {
            is_sep_byte(b)
        }
    }

    /// Extracts a slice corresponding to the portion of the path remaining for iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let mut components = Path::new("/tmp/foo/bar.txt").components();
    /// components.next();
    /// components.next();
    ///
    /// assert_eq!(Path::new("foo/bar.txt"), components.as_path());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_path(&self) -> &'a Path {
        let mut comps = self.clone();
        if comps.front == State::Body { comps.trim_left(); }
        if comps.back == State::Body { comps.trim_right(); }
        unsafe { Path::from_u8_slice(comps.path) }
    }

    /// Is the *original* path rooted?
    fn has_root(&self) -> bool {
        if self.has_physical_root { return true }
        if let Some(p) = self.prefix {
            if p.has_implicit_root() { return true }
        }
        false
    }

    /// Should the normalized path include a leading . ?
    fn include_cur_dir(&self) -> bool {
        if self.has_root() { return false }
        let mut iter = self.path[self.prefix_len()..].iter();
        match (iter.next(), iter.next()) {
            (Some(&b'.'), None) => true,
            (Some(&b'.'), Some(&b)) => self.is_sep_byte(b),
            _ => false
        }
    }

    // parse a given byte sequence into the corresponding path component
    fn parse_single_component<'b>(&self, comp: &'b [u8]) -> Option<Component<'b>> {
        match comp {
            b"." if self.prefix_verbatim() => Some(Component::CurDir),
            b"." => None, // . components are normalized away, except at
                          // the beginning of a path, which is treated
                          // separately via `include_cur_dir`
            b".." => Some(Component::ParentDir),
            b"" => None,
            _ => Some(Component::Normal(unsafe { u8_slice_as_os_str(comp) }))
        }
    }

    // parse a component from the left, saying how many bytes to consume to
    // remove the component
    fn parse_next_component(&self) -> (usize, Option<Component<'a>>) {
        debug_assert!(self.front == State::Body);
        let (extra, comp) = match self.path.iter().position(|b| self.is_sep_byte(*b)) {
            None => (0, self.path),
            Some(i) => (1, &self.path[.. i]),
        };
        (comp.len() + extra, self.parse_single_component(comp))
    }

    // parse a component from the right, saying how many bytes to consume to
    // remove the component
    fn parse_next_component_back(&self) -> (usize, Option<Component<'a>>) {
        debug_assert!(self.back == State::Body);
        let start = self.len_before_body();
        let (extra, comp) = match self.path[start..].iter().rposition(|b| self.is_sep_byte(*b)) {
            None => (0, &self.path[start ..]),
            Some(i) => (1, &self.path[start + i + 1 ..]),
        };
        (comp.len() + extra, self.parse_single_component(comp))
    }

    // trim away repeated separators (i.e. empty components) on the left
    fn trim_left(&mut self) {
        while !self.path.is_empty() {
            let (size, comp) = self.parse_next_component();
            if comp.is_some() {
                return;
            } else {
                self.path = &self.path[size ..];
            }
        }
    }

    // trim away repeated separators (i.e. empty components) on the right
    fn trim_right(&mut self) {
        while self.path.len() > self.len_before_body() {
            let (size, comp) = self.parse_next_component_back();
            if comp.is_some() {
                return;
            } else {
                self.path = &self.path[.. self.path.len() - size];
            }
        }
    }

    /// Examine the next component without consuming it.
    #[unstable(feature = "path_components_peek", issue = "27727")]
    pub fn peek(&self) -> Option<Component<'a>> {
        self.clone().next()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> AsRef<Path> for Components<'a> {
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> AsRef<OsStr> for Components<'a> {
    fn as_ref(&self) -> &OsStr {
        self.as_path().as_os_str()
    }
}

impl<'a> Iter<'a> {
    /// Extracts a slice corresponding to the portion of the path remaining for iteration.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_path(&self) -> &'a Path {
        self.inner.as_path()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> AsRef<Path> for Iter<'a> {
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> AsRef<OsStr> for Iter<'a> {
    fn as_ref(&self) -> &OsStr {
        self.as_path().as_os_str()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Iter<'a> {
    type Item = &'a OsStr;

    fn next(&mut self) -> Option<&'a OsStr> {
        self.inner.next().map(Component::as_os_str)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Iter<'a> {
    fn next_back(&mut self) -> Option<&'a OsStr> {
        self.inner.next_back().map(Component::as_os_str)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Components<'a> {
    type Item = Component<'a>;

    fn next(&mut self) -> Option<Component<'a>> {
        while !self.finished() {
            match self.front {
                State::Prefix if self.prefix_len() > 0 => {
                    self.front = State::StartDir;
                    debug_assert!(self.prefix_len() <= self.path.len());
                    let raw = &self.path[.. self.prefix_len()];
                    self.path = &self.path[self.prefix_len() .. ];
                    return Some(Component::Prefix(PrefixComponent {
                        raw: unsafe { u8_slice_as_os_str(raw) },
                        parsed: self.prefix.unwrap()
                    }))
                }
                State::Prefix => {
                    self.front = State::StartDir;
                }
                State::StartDir => {
                    self.front = State::Body;
                    if self.has_physical_root {
                        debug_assert!(!self.path.is_empty());
                        self.path = &self.path[1..];
                        return Some(Component::RootDir)
                    } else if let Some(p) = self.prefix {
                        if p.has_implicit_root() && !p.is_verbatim() {
                            return Some(Component::RootDir)
                        }
                    } else if self.include_cur_dir() {
                        debug_assert!(!self.path.is_empty());
                        self.path = &self.path[1..];
                        return Some(Component::CurDir)
                    }
                }
                State::Body if !self.path.is_empty() => {
                    let (size, comp) = self.parse_next_component();
                    self.path = &self.path[size ..];
                    if comp.is_some() { return comp }
                }
                State::Body => {
                    self.front = State::Done;
                }
                State::Done => unreachable!()
            }
        }
        None
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Components<'a> {
    fn next_back(&mut self) -> Option<Component<'a>> {
        while !self.finished() {
            match self.back {
                State::Body if self.path.len() > self.len_before_body() => {
                    let (size, comp) = self.parse_next_component_back();
                    self.path = &self.path[.. self.path.len() - size];
                    if comp.is_some() { return comp }
                }
                State::Body => {
                    self.back = State::StartDir;
                }
                State::StartDir => {
                    self.back = State::Prefix;
                    if self.has_physical_root {
                        self.path = &self.path[.. self.path.len() - 1];
                        return Some(Component::RootDir)
                    } else if let Some(p) = self.prefix {
                        if p.has_implicit_root() && !p.is_verbatim() {
                            return Some(Component::RootDir)
                        }
                    } else if self.include_cur_dir() {
                        self.path = &self.path[.. self.path.len() - 1];
                        return Some(Component::CurDir)
                    }
                }
                State::Prefix if self.prefix_len() > 0 => {
                    self.back = State::Done;
                    return Some(Component::Prefix(PrefixComponent {
                        raw: unsafe { u8_slice_as_os_str(self.path) },
                        parsed: self.prefix.unwrap()
                    }))
                }
                State::Prefix => {
                    self.back = State::Done;
                    return None
                }
                State::Done => unreachable!()
            }
        }
        None
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::PartialEq for Components<'a> {
    fn eq(&self, other: &Components<'a>) -> bool {
        Iterator::eq(self.clone(), other.clone())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::Eq for Components<'a> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::PartialOrd for Components<'a> {
    fn partial_cmp(&self, other: &Components<'a>) -> Option<cmp::Ordering> {
        Iterator::partial_cmp(self.clone(), other.clone())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::Ord for Components<'a> {
    fn cmp(&self, other: &Components<'a>) -> cmp::Ordering {
        Iterator::cmp(self.clone(), other.clone())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Basic types and traits
////////////////////////////////////////////////////////////////////////////////

/// An owned, mutable path (akin to `String`).
///
/// This type provides methods like `push` and `set_extension` that mutate the
/// path in place. It also implements `Deref` to `Path`, meaning that all
/// methods on `Path` slices are available on `PathBuf` values as well.
///
/// More details about the overall approach can be found in
/// the module documentation.
///
/// # Examples
///
/// ```
/// use std::path::PathBuf;
///
/// let mut path = PathBuf::from("c:\\");
/// path.push("windows");
/// path.push("system32");
/// path.set_extension("dll");
/// ```
#[derive(Clone, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct PathBuf {
    inner: OsString
}

impl PathBuf {
    fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        unsafe { &mut *(self as *mut PathBuf as *mut Vec<u8>) }
    }

    /// Allocates an empty `PathBuf`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> PathBuf {
        PathBuf { inner: OsString::new() }
    }

    /// Coerces to a `Path` slice.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_path(&self) -> &Path {
        self
    }

    /// Extends `self` with `path`.
    ///
    /// If `path` is absolute, it replaces the current path.
    ///
    /// On Windows:
    ///
    /// * if `path` has a root but no prefix (e.g. `\windows`), it
    ///   replaces everything except for the prefix (if any) of `self`.
    /// * if `path` has a prefix but no root, it replaces `self`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push<P: AsRef<Path>>(&mut self, path: P) {
        self._push(path.as_ref())
    }

    fn _push(&mut self, path: &Path) {
        // in general, a separator is needed if the rightmost byte is not a separator
        let mut need_sep = self.as_mut_vec().last().map(|c| !is_sep_byte(*c)).unwrap_or(false);

        // in the special case of `C:` on Windows, do *not* add a separator
        {
            let comps = self.components();
            if comps.prefix_len() > 0 &&
                comps.prefix_len() == comps.path.len() &&
                comps.prefix.unwrap().is_drive()
            {
                need_sep = false
            }
        }

        // absolute `path` replaces `self`
        if path.is_absolute() || path.prefix().is_some() {
            self.as_mut_vec().truncate(0);

        // `path` has a root but no prefix, e.g. `\windows` (Windows only)
        } else if path.has_root() {
            let prefix_len = self.components().prefix_remaining();
            self.as_mut_vec().truncate(prefix_len);

        // `path` is a pure relative path
        } else if need_sep {
            self.inner.push(MAIN_SEP_STR);
        }

        self.inner.push(path);
    }

    /// Truncate `self` to `self.parent()`.
    ///
    /// Returns false and does nothing if `self.file_name()` is `None`.
    /// Otherwise, returns `true`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop(&mut self) -> bool {
        match self.parent().map(|p| p.as_u8_slice().len()) {
            Some(len) => {
                self.as_mut_vec().truncate(len);
                true
            }
            None => false
        }
    }

    /// Updates `self.file_name()` to `file_name`.
    ///
    /// If `self.file_name()` was `None`, this is equivalent to pushing
    /// `file_name`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let mut buf = PathBuf::from("/");
    /// assert!(buf.file_name() == None);
    /// buf.set_file_name("bar");
    /// assert!(buf == PathBuf::from("/bar"));
    /// assert!(buf.file_name().is_some());
    /// buf.set_file_name("baz.txt");
    /// assert!(buf == PathBuf::from("/baz.txt"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set_file_name<S: AsRef<OsStr>>(&mut self, file_name: S) {
        self._set_file_name(file_name.as_ref())
    }

    fn _set_file_name(&mut self, file_name: &OsStr) {
        if self.file_name().is_some() {
            let popped = self.pop();
            debug_assert!(popped);
        }
        self.push(file_name);
    }

    /// Updates `self.extension()` to `extension`.
    ///
    /// If `self.file_name()` is `None`, does nothing and returns `false`.
    ///
    /// Otherwise, returns `true`; if `self.extension()` is `None`, the extension
    /// is added; otherwise it is replaced.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set_extension<S: AsRef<OsStr>>(&mut self, extension: S) -> bool {
        self._set_extension(extension.as_ref())
    }

    fn _set_extension(&mut self, extension: &OsStr) -> bool {
        if self.file_name().is_none() { return false; }

        let mut stem = match self.file_stem() {
            Some(stem) => stem.to_os_string(),
            None => OsString::new(),
        };

        if !os_str_as_u8_slice(extension).is_empty() {
            stem.push(".");
            stem.push(extension);
        }
        self.set_file_name(&stem);

        true
    }

    /// Consumes the `PathBuf`, yielding its internal `OsString` storage.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_os_string(self) -> OsString {
        self.inner
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized + AsRef<OsStr>> From<&'a T> for PathBuf {
    fn from(s: &'a T) -> PathBuf {
        PathBuf::from(s.as_ref().to_os_string())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<OsString> for PathBuf {
    fn from(s: OsString) -> PathBuf {
        PathBuf { inner: s }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<String> for PathBuf {
    fn from(s: String) -> PathBuf {
        PathBuf::from(OsString::from(s))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<P: AsRef<Path>> iter::FromIterator<P> for PathBuf {
    fn from_iter<I: IntoIterator<Item = P>>(iter: I) -> PathBuf {
        let mut buf = PathBuf::new();
        buf.extend(iter);
        buf
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<P: AsRef<Path>> iter::Extend<P> for PathBuf {
    fn extend<I: IntoIterator<Item = P>>(&mut self, iter: I) {
        for p in iter {
            self.push(p.as_ref())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for PathBuf {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(&**self, formatter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Deref for PathBuf {
    type Target = Path;

    fn deref(&self) -> &Path {
        Path::new(&self.inner)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Borrow<Path> for PathBuf {
    fn borrow(&self) -> &Path {
        self.deref()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl IntoCow<'static, Path> for PathBuf {
    fn into_cow(self) -> Cow<'static, Path> {
        Cow::Owned(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> IntoCow<'a, Path> for &'a Path {
    fn into_cow(self) -> Cow<'a, Path> {
        Cow::Borrowed(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToOwned for Path {
    type Owned = PathBuf;
    fn to_owned(&self) -> PathBuf { self.to_path_buf() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialEq for PathBuf {
    fn eq(&self, other: &PathBuf) -> bool {
        self.components() == other.components()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Eq for PathBuf {}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialOrd for PathBuf {
    fn partial_cmp(&self, other: &PathBuf) -> Option<cmp::Ordering> {
        self.components().partial_cmp(other.components())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Ord for PathBuf {
    fn cmp(&self, other: &PathBuf) -> cmp::Ordering {
        self.components().cmp(other.components())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for PathBuf {
    fn as_ref(&self) -> &OsStr {
        &self.inner[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Into<OsString> for PathBuf {
    fn into(self) -> OsString {
        self.inner
    }
}

/// A slice of a path (akin to `str`).
///
/// This type supports a number of operations for inspecting a path, including
/// breaking the path into its components (separated by `/` or `\`, depending on
/// the platform), extracting the file name, determining whether the path is
/// absolute, and so on. More details about the overall approach can be found in
/// the module documentation.
///
/// This is an *unsized* type, meaning that it must always be used behind a
/// pointer like `&` or `Box`.
///
/// # Examples
///
/// ```
/// use std::path::Path;
///
/// let path = Path::new("/tmp/foo/bar.txt");
/// let file = path.file_name();
/// let extension = path.extension();
/// let parent_dir = path.parent();
/// ```
///
#[derive(Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Path {
    inner: OsStr
}

impl Path {
    // The following (private!) function allows construction of a path from a u8
    // slice, which is only safe when it is known to follow the OsStr encoding.
    unsafe fn from_u8_slice(s: &[u8]) -> &Path {
        Path::new(u8_slice_as_os_str(s))
    }
    // The following (private!) function reveals the byte encoding used for OsStr.
    fn as_u8_slice(&self) -> &[u8] {
        os_str_as_u8_slice(&self.inner)
    }

    /// Directly wrap a string slice as a `Path` slice.
    ///
    /// This is a cost-free conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// Path::new("foo.txt");
    /// ```
    ///
    /// You can create `Path`s from `String`s, or even other `Path`s:
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let string = String::from("foo.txt");
    /// let from_string = Path::new(&string);
    /// let from_path = Path::new(&from_string);
    /// assert_eq!(from_string, from_path);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new<S: AsRef<OsStr> + ?Sized>(s: &S) -> &Path {
        unsafe { mem::transmute(s.as_ref()) }
    }

    /// Yields the underlying `OsStr` slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let os_str = Path::new("foo.txt").as_os_str();
    /// assert_eq!(os_str, std::ffi::OsStr::new("foo.txt"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_os_str(&self) -> &OsStr {
        &self.inner
    }

    /// Yields a `&str` slice if the `Path` is valid unicode.
    ///
    /// This conversion may entail doing a check for UTF-8 validity.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path_str = Path::new("foo.txt").to_str();
    /// assert_eq!(path_str, Some("foo.txt"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_str(&self) -> Option<&str> {
        self.inner.to_str()
    }

    /// Converts a `Path` to a `Cow<str>`.
    ///
    /// Any non-Unicode sequences are replaced with U+FFFD REPLACEMENT CHARACTER.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path_str = Path::new("foo.txt").to_string_lossy();
    /// assert_eq!(path_str, "foo.txt");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_string_lossy(&self) -> Cow<str> {
        self.inner.to_string_lossy()
    }

    /// Converts a `Path` to an owned `PathBuf`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path_buf = Path::new("foo.txt").to_path_buf();
    /// assert_eq!(path_buf, std::path::PathBuf::from("foo.txt"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_path_buf(&self) -> PathBuf {
        PathBuf::from(self.inner.to_os_string())
    }

    /// A path is *absolute* if it is independent of the current directory.
    ///
    /// * On Unix, a path is absolute if it starts with the root, so
    /// `is_absolute` and `has_root` are equivalent.
    ///
    /// * On Windows, a path is absolute if it has a prefix and starts with the
    /// root: `c:\windows` is absolute, while `c:temp` and `\temp` are not. In
    /// other words, `path.is_absolute() == path.prefix().is_some() && path.has_root()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// assert!(!Path::new("foo.txt").is_absolute());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_absolute(&self) -> bool {
        self.has_root() &&
            (cfg!(unix) || self.prefix().is_some())
    }

    /// A path is *relative* if it is not absolute.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// assert!(Path::new("foo.txt").is_relative());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_relative(&self) -> bool {
        !self.is_absolute()
    }

    /// Returns the *prefix* of a path, if any.
    ///
    /// Prefixes are relevant only for Windows paths, and consist of volumes
    /// like `C:`, UNC prefixes like `\\server`, and others described in more
    /// detail in `std::os::windows::PathExt`.
    #[unstable(feature = "path_prefix",
               reason = "uncertain whether to expose this convenience",
               issue = "27722")]
    pub fn prefix(&self) -> Option<Prefix> {
        self.components().prefix
    }

    /// A path has a root if the body of the path begins with the directory separator.
    ///
    /// * On Unix, a path has a root if it begins with `/`.
    ///
    /// * On Windows, a path has a root if it:
    ///     * has no prefix and begins with a separator, e.g. `\\windows`
    ///     * has a prefix followed by a separator, e.g. `c:\windows` but not `c:windows`
    ///     * has any non-disk prefix, e.g. `\\server\share`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// assert!(Path::new("/etc/passwd").has_root());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn has_root(&self) -> bool {
         self.components().has_root()
    }

    /// The path without its final component, if any.
    ///
    /// Returns `None` if the path terminates in a root or prefix.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("/foo/bar");
    /// let parent = path.parent().unwrap();
    /// assert_eq!(parent, Path::new("/foo"));
    ///
    /// let grand_parent = parent.parent().unwrap();
    /// assert_eq!(grand_parent, Path::new("/"));
    /// assert_eq!(grand_parent.parent(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn parent(&self) -> Option<&Path> {
        let mut comps = self.components();
        let comp = comps.next_back();
        comp.and_then(|p| match p {
            Component::Normal(_) |
            Component::CurDir |
            Component::ParentDir => Some(comps.as_path()),
            _ => None
        })
    }

    /// The final component of the path, if it is a normal file.
    ///
    /// If the path terminates in `.`, `..`, or consists solely of a root of
    /// prefix, `file_name` will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use std::ffi::OsStr;
    ///
    /// let path = Path::new("foo.txt");
    /// let os_str = OsStr::new("foo.txt");
    ///
    /// assert_eq!(Some(os_str), path.file_name());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn file_name(&self) -> Option<&OsStr> {
        self.components().next_back().and_then(|p| match p {
            Component::Normal(p) => Some(p.as_ref()),
            _ => None
        })
    }

    /// Returns a path that, when joined onto `base`, yields `self`.
    ///
    /// If `base` is not a prefix of `self` (i.e. `starts_with`
    /// returns false), then `relative_from` returns `None`.
    #[unstable(feature = "path_relative_from", reason = "see #23284",
               issue = "23284")]
    pub fn relative_from<'a, P: ?Sized + AsRef<Path>>(&'a self, base: &'a P) -> Option<&Path>
    {
        self._relative_from(base.as_ref())
    }

    fn _relative_from<'a>(&'a self, base: &'a Path) -> Option<&'a Path> {
        iter_after(self.components(), base.components()).map(|c| c.as_path())
    }

    /// Determines whether `base` is a prefix of `self`.
    ///
    /// Only considers whole path components to match.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("/etc/passwd");
    ///
    /// assert!(path.starts_with("/etc"));
    ///
    /// assert!(!path.starts_with("/e"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn starts_with<P: AsRef<Path>>(&self, base: P) -> bool {
        self._starts_with(base.as_ref())
    }

    fn _starts_with(&self, base: &Path) -> bool {
        iter_after(self.components(), base.components()).is_some()
    }

    /// Determines whether `child` is a suffix of `self`.
    ///
    /// Only considers whole path components to match.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("/etc/passwd");
    ///
    /// assert!(path.ends_with("passwd"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ends_with<P: AsRef<Path>>(&self, child: P) -> bool {
        self._ends_with(child.as_ref())
    }

    fn _ends_with(&self, child: &Path) -> bool {
        iter_after(self.components().rev(), child.components().rev()).is_some()
    }

    /// Extracts the stem (non-extension) portion of `self.file_name()`.
    ///
    /// The stem is:
    ///
    /// * None, if there is no file name;
    /// * The entire file name if there is no embedded `.`;
    /// * The entire file name if the file name begins with `.` and has no other `.`s within;
    /// * Otherwise, the portion of the file name before the final `.`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("foo.rs");
    ///
    /// assert_eq!("foo", path.file_stem().unwrap());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn file_stem(&self) -> Option<&OsStr> {
        self.file_name().map(split_file_at_dot).and_then(|(before, after)| before.or(after))
    }

    /// Extracts the extension of `self.file_name()`, if possible.
    ///
    /// The extension is:
    ///
    /// * None, if there is no file name;
    /// * None, if there is no embedded `.`;
    /// * None, if the file name begins with `.` and has no other `.`s within;
    /// * Otherwise, the portion of the file name after the final `.`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("foo.rs");
    ///
    /// assert_eq!("rs", path.extension().unwrap());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn extension(&self) -> Option<&OsStr> {
        self.file_name().map(split_file_at_dot).and_then(|(before, after)| before.and(after))
    }

    /// Creates an owned `PathBuf` with `path` adjoined to `self`.
    ///
    /// See `PathBuf::push` for more details on what it means to adjoin a path.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// assert_eq!(Path::new("/etc").join("passwd"), PathBuf::from("/etc/passwd"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn join<P: AsRef<Path>>(&self, path: P) -> PathBuf {
        self._join(path.as_ref())
    }

    fn _join(&self, path: &Path) -> PathBuf {
        let mut buf = self.to_path_buf();
        buf.push(path);
        buf
    }

    /// Creates an owned `PathBuf` like `self` but with the given file name.
    ///
    /// See `PathBuf::set_file_name` for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let path = Path::new("/tmp/foo.txt");
    /// assert_eq!(path.with_file_name("bar.txt"), PathBuf::from("/tmp/bar.txt"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_file_name<S: AsRef<OsStr>>(&self, file_name: S) -> PathBuf {
        self._with_file_name(file_name.as_ref())
    }

    fn _with_file_name(&self, file_name: &OsStr) -> PathBuf {
        let mut buf = self.to_path_buf();
        buf.set_file_name(file_name);
        buf
    }

    /// Creates an owned `PathBuf` like `self` but with the given extension.
    ///
    /// See `PathBuf::set_extension` for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let path = Path::new("foo.rs");
    /// assert_eq!(path.with_extension("txt"), PathBuf::from("foo.txt"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_extension<S: AsRef<OsStr>>(&self, extension: S) -> PathBuf {
        self._with_extension(extension.as_ref())
    }

    fn _with_extension(&self, extension: &OsStr) -> PathBuf {
        let mut buf = self.to_path_buf();
        buf.set_extension(extension);
        buf
    }

    /// Produce an iterator over the components of the path.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, Component};
    /// use std::ffi::OsStr;
    ///
    /// let mut components = Path::new("/tmp/foo.txt").components();
    ///
    /// assert_eq!(components.next(), Some(Component::RootDir));
    /// assert_eq!(components.next(), Some(Component::Normal(OsStr::new("tmp"))));
    /// assert_eq!(components.next(), Some(Component::Normal(OsStr::new("foo.txt"))));
    /// assert_eq!(components.next(), None)
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn components(&self) -> Components {
        let prefix = parse_prefix(self.as_os_str());
        Components {
            path: self.as_u8_slice(),
            prefix: prefix,
            has_physical_root: has_physical_root(self.as_u8_slice(), prefix),
            front: State::Prefix,
            back: State::Body,
        }
    }

    /// Produce an iterator over the path's components viewed as `OsStr` slices.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{self, Path};
    /// use std::ffi::OsStr;
    ///
    /// let mut it = Path::new("/tmp/foo.txt").iter();
    /// assert_eq!(it.next(), Some(OsStr::new(&path::MAIN_SEPARATOR.to_string())));
    /// assert_eq!(it.next(), Some(OsStr::new("tmp")));
    /// assert_eq!(it.next(), Some(OsStr::new("foo.txt")));
    /// assert_eq!(it.next(), None)
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter {
        Iter { inner: self.components() }
    }

    /// Returns an object that implements `Display` for safely printing paths
    /// that may contain non-Unicode data.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("/tmp/foo.rs");
    ///
    /// println!("{}", path.display());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn display(&self) -> Display {
        Display { path: self }
    }


    /// Gets information on the file, directory, etc at this path.
    ///
    /// Consult the `fs::metadata` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with
    /// `fs::metadata`.
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn metadata(&self) -> io::Result<fs::Metadata> {
        fs::metadata(self)
    }

    /// Gets information on the file, directory, etc at this path.
    ///
    /// Consult the `fs::symlink_metadata` documentation for more info.
    ///
    /// This call preserves identical runtime/error semantics with
    /// `fs::symlink_metadata`.
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn symlink_metadata(&self) -> io::Result<fs::Metadata> {
        fs::symlink_metadata(self)
    }

    /// Returns the canonical form of a path, normalizing all components and
    /// eliminate all symlinks.
    ///
    /// This call preserves identical runtime/error semantics with
    /// `fs::canonicalize`.
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn canonicalize(&self) -> io::Result<PathBuf> {
        fs::canonicalize(self)
    }

    /// Reads the symlink at this path.
    ///
    /// For more information see `fs::read_link`.
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn read_link(&self) -> io::Result<PathBuf> {
        fs::read_link(self)
    }

    /// Reads the directory at this path.
    ///
    /// For more information see `fs::read_dir`.
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn read_dir(&self) -> io::Result<fs::ReadDir> {
        fs::read_dir(self)
    }

    /// Boolean value indicator whether the underlying file exists on the local
    /// filesystem. Returns false in exactly the cases where `fs::stat` fails.
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn exists(&self) -> bool {
        fs::metadata(self).is_ok()
    }

    /// Whether the underlying implementation (be it a file path, or something
    /// else) points at a "regular file" on the FS. Will return false for paths
    /// to non-existent locations or directories or other non-regular files
    /// (named pipes, etc). Follows links when making this determination.
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn is_file(&self) -> bool {
        fs::metadata(self).map(|m| m.is_file()).unwrap_or(false)
    }

    /// Whether the underlying implementation (be it a file path, or something
    /// else) is pointing at a directory in the underlying FS. Will return
    /// false for paths to non-existent locations or if the item is not a
    /// directory (eg files, named pipes, etc). Follows links when making this
    /// determination.
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn is_dir(&self) -> bool {
        fs::metadata(self).map(|m| m.is_dir()).unwrap_or(false)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for Path {
    fn as_ref(&self) -> &OsStr {
        &self.inner
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Path {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.inner.fmt(formatter)
    }
}

/// Helper struct for safely printing paths with `format!()` and `{}`
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Display<'a> {
    path: &'a Path
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> fmt::Debug for Display<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.path.to_string_lossy(), f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> fmt::Display for Display<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.path.to_string_lossy(), f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialEq for Path {
    fn eq(&self, other: &Path) -> bool {
        self.components().eq(other.components())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Eq for Path {}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialOrd for Path {
    fn partial_cmp(&self, other: &Path) -> Option<cmp::Ordering> {
        self.components().partial_cmp(other.components())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Ord for Path {
    fn cmp(&self, other: &Path) -> cmp::Ordering {
        self.components().cmp(other.components())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for Path {
    fn as_ref(&self) -> &Path { self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for OsStr {
    fn as_ref(&self) -> &Path { Path::new(self) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for OsString {
    fn as_ref(&self) -> &Path { Path::new(self) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for str {
    fn as_ref(&self) -> &Path { Path::new(self) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for String {
    fn as_ref(&self) -> &Path { Path::new(self) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for PathBuf {
    fn as_ref(&self) -> &Path { self }
}

#[stable(feature = "path_into_iter", since = "1.6.0")]
impl<'a> IntoIterator for &'a PathBuf {
    type Item = &'a OsStr;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> { self.iter() }
}

#[stable(feature = "path_into_iter", since = "1.6.0")]
impl<'a> IntoIterator for &'a Path {
    type Item = &'a OsStr;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> { self.iter() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use string::{ToString, String};
    use vec::Vec;

    macro_rules! t(
        ($path:expr, iter: $iter:expr) => (
            {
                let path = Path::new($path);

                // Forward iteration
                let comps = path.iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect::<Vec<String>>();
                let exp: &[&str] = &$iter;
                let exps = exp.iter().map(|s| s.to_string()).collect::<Vec<String>>();
                assert!(comps == exps, "iter: Expected {:?}, found {:?}",
                        exps, comps);

                // Reverse iteration
                let comps = Path::new($path).iter().rev()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect::<Vec<String>>();
                let exps = exps.into_iter().rev().collect::<Vec<String>>();
                assert!(comps == exps, "iter().rev(): Expected {:?}, found {:?}",
                        exps, comps);
            }
        );

        ($path:expr, has_root: $has_root:expr, is_absolute: $is_absolute:expr) => (
            {
                let path = Path::new($path);

                let act_root = path.has_root();
                assert!(act_root == $has_root, "has_root: Expected {:?}, found {:?}",
                        $has_root, act_root);

                let act_abs = path.is_absolute();
                assert!(act_abs == $is_absolute, "is_absolute: Expected {:?}, found {:?}",
                        $is_absolute, act_abs);
            }
        );

        ($path:expr, parent: $parent:expr, file_name: $file:expr) => (
            {
                let path = Path::new($path);

                let parent = path.parent().map(|p| p.to_str().unwrap());
                let exp_parent: Option<&str> = $parent;
                assert!(parent == exp_parent, "parent: Expected {:?}, found {:?}",
                        exp_parent, parent);

                let file = path.file_name().map(|p| p.to_str().unwrap());
                let exp_file: Option<&str> = $file;
                assert!(file == exp_file, "file_name: Expected {:?}, found {:?}",
                        exp_file, file);
            }
        );

        ($path:expr, file_stem: $file_stem:expr, extension: $extension:expr) => (
            {
                let path = Path::new($path);

                let stem = path.file_stem().map(|p| p.to_str().unwrap());
                let exp_stem: Option<&str> = $file_stem;
                assert!(stem == exp_stem, "file_stem: Expected {:?}, found {:?}",
                        exp_stem, stem);

                let ext = path.extension().map(|p| p.to_str().unwrap());
                let exp_ext: Option<&str> = $extension;
                assert!(ext == exp_ext, "extension: Expected {:?}, found {:?}",
                        exp_ext, ext);
            }
        );

        ($path:expr, iter: $iter:expr,
                     has_root: $has_root:expr, is_absolute: $is_absolute:expr,
                     parent: $parent:expr, file_name: $file:expr,
                     file_stem: $file_stem:expr, extension: $extension:expr) => (
            {
                t!($path, iter: $iter);
                t!($path, has_root: $has_root, is_absolute: $is_absolute);
                t!($path, parent: $parent, file_name: $file);
                t!($path, file_stem: $file_stem, extension: $extension);
            }
        );
    );

    #[test]
    fn into_cow() {
        use borrow::{Cow, IntoCow};

        let static_path = Path::new("/home/foo");
        let static_cow_path: Cow<'static, Path> = static_path.into_cow();
        let pathbuf = PathBuf::from("/home/foo");

        {
            let path: &Path = &pathbuf;
            let borrowed_cow_path: Cow<Path> = path.into_cow();

            assert_eq!(static_cow_path, borrowed_cow_path);
        }

        let owned_cow_path: Cow<'static, Path> = pathbuf.into_cow();

        assert_eq!(static_cow_path, owned_cow_path);
    }

    #[test]
    #[cfg(unix)]
    pub fn test_decompositions_unix() {
        t!("",
           iter: [],
           has_root: false,
           is_absolute: false,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("foo",
           iter: ["foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("/",
           iter: ["/"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("/foo",
           iter: ["/", "foo"],
           has_root: true,
           is_absolute: true,
           parent: Some("/"),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo/",
           iter: ["foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("/foo/",
           iter: ["/", "foo"],
           has_root: true,
           is_absolute: true,
           parent: Some("/"),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo/bar",
           iter: ["foo", "bar"],
           has_root: false,
           is_absolute: false,
           parent: Some("foo"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("/foo/bar",
           iter: ["/", "foo", "bar"],
           has_root: true,
           is_absolute: true,
           parent: Some("/foo"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("///foo///",
           iter: ["/", "foo"],
           has_root: true,
           is_absolute: true,
           parent: Some("/"),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("///foo///bar",
           iter: ["/", "foo", "bar"],
           has_root: true,
           is_absolute: true,
           parent: Some("///foo"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("./.",
           iter: ["."],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("/..",
           iter: ["/", ".."],
           has_root: true,
           is_absolute: true,
           parent: Some("/"),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("../",
           iter: [".."],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("foo/.",
           iter: ["foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo/..",
           iter: ["foo", ".."],
           has_root: false,
           is_absolute: false,
           parent: Some("foo"),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("foo/./",
           iter: ["foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo/./bar",
           iter: ["foo", "bar"],
           has_root: false,
           is_absolute: false,
           parent: Some("foo"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("foo/../",
           iter: ["foo", ".."],
           has_root: false,
           is_absolute: false,
           parent: Some("foo"),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("foo/../bar",
           iter: ["foo", "..", "bar"],
           has_root: false,
           is_absolute: false,
           parent: Some("foo/.."),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("./a",
           iter: [".", "a"],
           has_root: false,
           is_absolute: false,
           parent: Some("."),
           file_name: Some("a"),
           file_stem: Some("a"),
           extension: None
           );

        t!(".",
           iter: ["."],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("./",
           iter: ["."],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("a/b",
           iter: ["a", "b"],
           has_root: false,
           is_absolute: false,
           parent: Some("a"),
           file_name: Some("b"),
           file_stem: Some("b"),
           extension: None
           );

        t!("a//b",
           iter: ["a", "b"],
           has_root: false,
           is_absolute: false,
           parent: Some("a"),
           file_name: Some("b"),
           file_stem: Some("b"),
           extension: None
           );

        t!("a/./b",
           iter: ["a", "b"],
           has_root: false,
           is_absolute: false,
           parent: Some("a"),
           file_name: Some("b"),
           file_stem: Some("b"),
           extension: None
           );

        t!("a/b/c",
           iter: ["a", "b", "c"],
           has_root: false,
           is_absolute: false,
           parent: Some("a/b"),
           file_name: Some("c"),
           file_stem: Some("c"),
           extension: None
           );

        t!(".foo",
           iter: [".foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some(".foo"),
           file_stem: Some(".foo"),
           extension: None
           );
    }

    #[test]
    #[cfg(windows)]
    pub fn test_decompositions_windows() {
        t!("",
           iter: [],
           has_root: false,
           is_absolute: false,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("foo",
           iter: ["foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("/",
           iter: ["\\"],
           has_root: true,
           is_absolute: false,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("\\",
           iter: ["\\"],
           has_root: true,
           is_absolute: false,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("c:",
           iter: ["c:"],
           has_root: false,
           is_absolute: false,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("c:\\",
           iter: ["c:", "\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("c:/",
           iter: ["c:", "\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("/foo",
           iter: ["\\", "foo"],
           has_root: true,
           is_absolute: false,
           parent: Some("/"),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo/",
           iter: ["foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("/foo/",
           iter: ["\\", "foo"],
           has_root: true,
           is_absolute: false,
           parent: Some("/"),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo/bar",
           iter: ["foo", "bar"],
           has_root: false,
           is_absolute: false,
           parent: Some("foo"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("/foo/bar",
           iter: ["\\", "foo", "bar"],
           has_root: true,
           is_absolute: false,
           parent: Some("/foo"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("///foo///",
           iter: ["\\", "foo"],
           has_root: true,
           is_absolute: false,
           parent: Some("/"),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("///foo///bar",
           iter: ["\\", "foo", "bar"],
           has_root: true,
           is_absolute: false,
           parent: Some("///foo"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("./.",
           iter: ["."],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("/..",
           iter: ["\\", ".."],
           has_root: true,
           is_absolute: false,
           parent: Some("/"),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("../",
           iter: [".."],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("foo/.",
           iter: ["foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo/..",
           iter: ["foo", ".."],
           has_root: false,
           is_absolute: false,
           parent: Some("foo"),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("foo/./",
           iter: ["foo"],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: Some("foo"),
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo/./bar",
           iter: ["foo", "bar"],
           has_root: false,
           is_absolute: false,
           parent: Some("foo"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("foo/../",
           iter: ["foo", ".."],
           has_root: false,
           is_absolute: false,
           parent: Some("foo"),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("foo/../bar",
           iter: ["foo", "..", "bar"],
           has_root: false,
           is_absolute: false,
           parent: Some("foo/.."),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );

        t!("./a",
           iter: [".", "a"],
           has_root: false,
           is_absolute: false,
           parent: Some("."),
           file_name: Some("a"),
           file_stem: Some("a"),
           extension: None
           );

        t!(".",
           iter: ["."],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("./",
           iter: ["."],
           has_root: false,
           is_absolute: false,
           parent: Some(""),
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("a/b",
           iter: ["a", "b"],
           has_root: false,
           is_absolute: false,
           parent: Some("a"),
           file_name: Some("b"),
           file_stem: Some("b"),
           extension: None
           );

        t!("a//b",
           iter: ["a", "b"],
           has_root: false,
           is_absolute: false,
           parent: Some("a"),
           file_name: Some("b"),
           file_stem: Some("b"),
           extension: None
           );

        t!("a/./b",
           iter: ["a", "b"],
           has_root: false,
           is_absolute: false,
           parent: Some("a"),
           file_name: Some("b"),
           file_stem: Some("b"),
           extension: None
           );

        t!("a/b/c",
           iter: ["a", "b", "c"],
           has_root: false,
           is_absolute: false,
           parent: Some("a/b"),
           file_name: Some("c"),
           file_stem: Some("c"),
           extension: None);

        t!("a\\b\\c",
           iter: ["a", "b", "c"],
           has_root: false,
           is_absolute: false,
           parent: Some("a\\b"),
           file_name: Some("c"),
           file_stem: Some("c"),
           extension: None
           );

        t!("\\a",
           iter: ["\\", "a"],
           has_root: true,
           is_absolute: false,
           parent: Some("\\"),
           file_name: Some("a"),
           file_stem: Some("a"),
           extension: None
           );

        t!("c:\\foo.txt",
           iter: ["c:", "\\", "foo.txt"],
           has_root: true,
           is_absolute: true,
           parent: Some("c:\\"),
           file_name: Some("foo.txt"),
           file_stem: Some("foo"),
           extension: Some("txt")
           );

        t!("\\\\server\\share\\foo.txt",
           iter: ["\\\\server\\share", "\\", "foo.txt"],
           has_root: true,
           is_absolute: true,
           parent: Some("\\\\server\\share\\"),
           file_name: Some("foo.txt"),
           file_stem: Some("foo"),
           extension: Some("txt")
           );

        t!("\\\\server\\share",
           iter: ["\\\\server\\share", "\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("\\\\server",
           iter: ["\\", "server"],
           has_root: true,
           is_absolute: false,
           parent: Some("\\"),
           file_name: Some("server"),
           file_stem: Some("server"),
           extension: None
           );

        t!("\\\\?\\bar\\foo.txt",
           iter: ["\\\\?\\bar", "\\", "foo.txt"],
           has_root: true,
           is_absolute: true,
           parent: Some("\\\\?\\bar\\"),
           file_name: Some("foo.txt"),
           file_stem: Some("foo"),
           extension: Some("txt")
           );

        t!("\\\\?\\bar",
           iter: ["\\\\?\\bar"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("\\\\?\\",
           iter: ["\\\\?\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("\\\\?\\UNC\\server\\share\\foo.txt",
           iter: ["\\\\?\\UNC\\server\\share", "\\", "foo.txt"],
           has_root: true,
           is_absolute: true,
           parent: Some("\\\\?\\UNC\\server\\share\\"),
           file_name: Some("foo.txt"),
           file_stem: Some("foo"),
           extension: Some("txt")
           );

        t!("\\\\?\\UNC\\server",
           iter: ["\\\\?\\UNC\\server"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("\\\\?\\UNC\\",
           iter: ["\\\\?\\UNC\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("\\\\?\\C:\\foo.txt",
           iter: ["\\\\?\\C:", "\\", "foo.txt"],
           has_root: true,
           is_absolute: true,
           parent: Some("\\\\?\\C:\\"),
           file_name: Some("foo.txt"),
           file_stem: Some("foo"),
           extension: Some("txt")
           );


        t!("\\\\?\\C:\\",
           iter: ["\\\\?\\C:", "\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );


        t!("\\\\?\\C:",
           iter: ["\\\\?\\C:"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );


        t!("\\\\?\\foo/bar",
           iter: ["\\\\?\\foo/bar"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );


        t!("\\\\?\\C:/foo",
           iter: ["\\\\?\\C:/foo"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );


        t!("\\\\.\\foo\\bar",
           iter: ["\\\\.\\foo", "\\", "bar"],
           has_root: true,
           is_absolute: true,
           parent: Some("\\\\.\\foo\\"),
           file_name: Some("bar"),
           file_stem: Some("bar"),
           extension: None
           );


        t!("\\\\.\\foo",
           iter: ["\\\\.\\foo", "\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );


        t!("\\\\.\\foo/bar",
           iter: ["\\\\.\\foo/bar", "\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );


        t!("\\\\.\\foo\\bar/baz",
           iter: ["\\\\.\\foo", "\\", "bar", "baz"],
           has_root: true,
           is_absolute: true,
           parent: Some("\\\\.\\foo\\bar"),
           file_name: Some("baz"),
           file_stem: Some("baz"),
           extension: None
           );


        t!("\\\\.\\",
           iter: ["\\\\.\\", "\\"],
           has_root: true,
           is_absolute: true,
           parent: None,
           file_name: None,
           file_stem: None,
           extension: None
           );

        t!("\\\\?\\a\\b\\",
           iter: ["\\\\?\\a", "\\", "b"],
           has_root: true,
           is_absolute: true,
           parent: Some("\\\\?\\a\\"),
           file_name: Some("b"),
           file_stem: Some("b"),
           extension: None
           );
    }

    #[test]
    pub fn test_stem_ext() {
        t!("foo",
           file_stem: Some("foo"),
           extension: None
           );

        t!("foo.",
           file_stem: Some("foo"),
           extension: Some("")
           );

        t!(".foo",
           file_stem: Some(".foo"),
           extension: None
           );

        t!("foo.txt",
           file_stem: Some("foo"),
           extension: Some("txt")
           );

        t!("foo.bar.txt",
           file_stem: Some("foo.bar"),
           extension: Some("txt")
           );

        t!("foo.bar.",
           file_stem: Some("foo.bar"),
           extension: Some("")
           );

        t!(".",
           file_stem: None,
           extension: None
           );

        t!("..",
           file_stem: None,
           extension: None
           );

        t!("",
           file_stem: None,
           extension: None
           );
    }

    #[test]
    pub fn test_push() {
        macro_rules! tp(
            ($path:expr, $push:expr, $expected:expr) => ( {
                let mut actual = PathBuf::from($path);
                actual.push($push);
                assert!(actual.to_str() == Some($expected),
                        "pushing {:?} onto {:?}: Expected {:?}, got {:?}",
                        $push, $path, $expected, actual.to_str().unwrap());
            });
        );

        if cfg!(unix) {
            tp!("", "foo", "foo");
            tp!("foo", "bar", "foo/bar");
            tp!("foo/", "bar", "foo/bar");
            tp!("foo//", "bar", "foo//bar");
            tp!("foo/.", "bar", "foo/./bar");
            tp!("foo./.", "bar", "foo././bar");
            tp!("foo", "", "foo/");
            tp!("foo", ".", "foo/.");
            tp!("foo", "..", "foo/..");
            tp!("foo", "/", "/");
            tp!("/foo/bar", "/", "/");
            tp!("/foo/bar", "/baz", "/baz");
            tp!("/foo/bar", "./baz", "/foo/bar/./baz");
        } else {
            tp!("", "foo", "foo");
            tp!("foo", "bar", r"foo\bar");
            tp!("foo/", "bar", r"foo/bar");
            tp!(r"foo\", "bar", r"foo\bar");
            tp!("foo//", "bar", r"foo//bar");
            tp!(r"foo\\", "bar", r"foo\\bar");
            tp!("foo/.", "bar", r"foo/.\bar");
            tp!("foo./.", "bar", r"foo./.\bar");
            tp!(r"foo\.", "bar", r"foo\.\bar");
            tp!(r"foo.\.", "bar", r"foo.\.\bar");
            tp!("foo", "", "foo\\");
            tp!("foo", ".", r"foo\.");
            tp!("foo", "..", r"foo\..");
            tp!("foo", "/", "/");
            tp!("foo", r"\", r"\");
            tp!("/foo/bar", "/", "/");
            tp!(r"\foo\bar", r"\", r"\");
            tp!("/foo/bar", "/baz", "/baz");
            tp!("/foo/bar", r"\baz", r"\baz");
            tp!("/foo/bar", "./baz", r"/foo/bar\./baz");
            tp!("/foo/bar", r".\baz", r"/foo/bar\.\baz");

            tp!("c:\\", "windows", "c:\\windows");
            tp!("c:", "windows", "c:windows");

            tp!("a\\b\\c", "d", "a\\b\\c\\d");
            tp!("\\a\\b\\c", "d", "\\a\\b\\c\\d");
            tp!("a\\b", "c\\d", "a\\b\\c\\d");
            tp!("a\\b", "\\c\\d", "\\c\\d");
            tp!("a\\b", ".", "a\\b\\.");
            tp!("a\\b", "..\\c", "a\\b\\..\\c");
            tp!("a\\b", "C:a.txt", "C:a.txt");
            tp!("a\\b", "C:\\a.txt", "C:\\a.txt");
            tp!("C:\\a", "C:\\b.txt", "C:\\b.txt");
            tp!("C:\\a\\b\\c", "C:d", "C:d");
            tp!("C:a\\b\\c", "C:d", "C:d");
            tp!("C:", r"a\b\c", r"C:a\b\c");
            tp!("C:", r"..\a", r"C:..\a");
            tp!("\\\\server\\share\\foo", "bar", "\\\\server\\share\\foo\\bar");
            tp!("\\\\server\\share\\foo", "C:baz", "C:baz");
            tp!("\\\\?\\C:\\a\\b", "C:c\\d", "C:c\\d");
            tp!("\\\\?\\C:a\\b", "C:c\\d", "C:c\\d");
            tp!("\\\\?\\C:\\a\\b", "C:\\c\\d", "C:\\c\\d");
            tp!("\\\\?\\foo\\bar", "baz", "\\\\?\\foo\\bar\\baz");
            tp!("\\\\?\\UNC\\server\\share\\foo", "bar", "\\\\?\\UNC\\server\\share\\foo\\bar");
            tp!("\\\\?\\UNC\\server\\share", "C:\\a", "C:\\a");
            tp!("\\\\?\\UNC\\server\\share", "C:a", "C:a");

            // Note: modified from old path API
            tp!("\\\\?\\UNC\\server", "foo", "\\\\?\\UNC\\server\\foo");

            tp!("C:\\a", "\\\\?\\UNC\\server\\share", "\\\\?\\UNC\\server\\share");
            tp!("\\\\.\\foo\\bar", "baz", "\\\\.\\foo\\bar\\baz");
            tp!("\\\\.\\foo\\bar", "C:a", "C:a");
            // again, not sure about the following, but I'm assuming \\.\ should be verbatim
            tp!("\\\\.\\foo", "..\\bar", "\\\\.\\foo\\..\\bar");

            tp!("\\\\?\\C:", "foo", "\\\\?\\C:\\foo"); // this is a weird one
        }
    }

    #[test]
    pub fn test_pop() {
        macro_rules! tp(
            ($path:expr, $expected:expr, $output:expr) => ( {
                let mut actual = PathBuf::from($path);
                let output = actual.pop();
                assert!(actual.to_str() == Some($expected) && output == $output,
                        "popping from {:?}: Expected {:?}/{:?}, got {:?}/{:?}",
                        $path, $expected, $output,
                        actual.to_str().unwrap(), output);
            });
        );

        tp!("", "", false);
        tp!("/", "/", false);
        tp!("foo", "", true);
        tp!(".", "", true);
        tp!("/foo", "/", true);
        tp!("/foo/bar", "/foo", true);
        tp!("foo/bar", "foo", true);
        tp!("foo/.", "", true);
        tp!("foo//bar", "foo", true);

        if cfg!(windows) {
            tp!("a\\b\\c", "a\\b", true);
            tp!("\\a", "\\", true);
            tp!("\\", "\\", false);

            tp!("C:\\a\\b", "C:\\a", true);
            tp!("C:\\a", "C:\\", true);
            tp!("C:\\", "C:\\", false);
            tp!("C:a\\b", "C:a", true);
            tp!("C:a", "C:", true);
            tp!("C:", "C:", false);
            tp!("\\\\server\\share\\a\\b", "\\\\server\\share\\a", true);
            tp!("\\\\server\\share\\a", "\\\\server\\share\\", true);
            tp!("\\\\server\\share", "\\\\server\\share", false);
            tp!("\\\\?\\a\\b\\c", "\\\\?\\a\\b", true);
            tp!("\\\\?\\a\\b", "\\\\?\\a\\", true);
            tp!("\\\\?\\a", "\\\\?\\a", false);
            tp!("\\\\?\\C:\\a\\b", "\\\\?\\C:\\a", true);
            tp!("\\\\?\\C:\\a", "\\\\?\\C:\\", true);
            tp!("\\\\?\\C:\\", "\\\\?\\C:\\", false);
            tp!("\\\\?\\UNC\\server\\share\\a\\b", "\\\\?\\UNC\\server\\share\\a", true);
            tp!("\\\\?\\UNC\\server\\share\\a", "\\\\?\\UNC\\server\\share\\", true);
            tp!("\\\\?\\UNC\\server\\share", "\\\\?\\UNC\\server\\share", false);
            tp!("\\\\.\\a\\b\\c", "\\\\.\\a\\b", true);
            tp!("\\\\.\\a\\b", "\\\\.\\a\\", true);
            tp!("\\\\.\\a", "\\\\.\\a", false);

            tp!("\\\\?\\a\\b\\", "\\\\?\\a\\", true);
        }
    }

    #[test]
    pub fn test_set_file_name() {
        macro_rules! tfn(
                ($path:expr, $file:expr, $expected:expr) => ( {
                let mut p = PathBuf::from($path);
                p.set_file_name($file);
                assert!(p.to_str() == Some($expected),
                        "setting file name of {:?} to {:?}: Expected {:?}, got {:?}",
                        $path, $file, $expected,
                        p.to_str().unwrap());
            });
        );

        tfn!("foo", "foo", "foo");
        tfn!("foo", "bar", "bar");
        tfn!("foo", "", "");
        tfn!("", "foo", "foo");
        if cfg!(unix) {
            tfn!(".", "foo", "./foo");
            tfn!("foo/", "bar", "bar");
            tfn!("foo/.", "bar", "bar");
            tfn!("..", "foo", "../foo");
            tfn!("foo/..", "bar", "foo/../bar");
            tfn!("/", "foo", "/foo");
        } else {
            tfn!(".", "foo", r".\foo");
            tfn!(r"foo\", "bar", r"bar");
            tfn!(r"foo\.", "bar", r"bar");
            tfn!("..", "foo", r"..\foo");
            tfn!(r"foo\..", "bar", r"foo\..\bar");
            tfn!(r"\", "foo", r"\foo");
        }
    }

    #[test]
    pub fn test_set_extension() {
        macro_rules! tfe(
                ($path:expr, $ext:expr, $expected:expr, $output:expr) => ( {
                let mut p = PathBuf::from($path);
                let output = p.set_extension($ext);
                assert!(p.to_str() == Some($expected) && output == $output,
                        "setting extension of {:?} to {:?}: Expected {:?}/{:?}, got {:?}/{:?}",
                        $path, $ext, $expected, $output,
                        p.to_str().unwrap(), output);
            });
        );

        tfe!("foo", "txt", "foo.txt", true);
        tfe!("foo.bar", "txt", "foo.txt", true);
        tfe!("foo.bar.baz", "txt", "foo.bar.txt", true);
        tfe!(".test", "txt", ".test.txt", true);
        tfe!("foo.txt", "", "foo", true);
        tfe!("foo", "", "foo", true);
        tfe!("", "foo", "", false);
        tfe!(".", "foo", ".", false);
        tfe!("foo/", "bar", "foo.bar", true);
        tfe!("foo/.", "bar", "foo.bar", true);
        tfe!("..", "foo", "..",  false);
        tfe!("foo/..", "bar", "foo/..", false);
        tfe!("/", "foo", "/", false);
    }

    #[test]
    pub fn test_compare() {
        macro_rules! tc(
            ($path1:expr, $path2:expr, eq: $eq:expr,
             starts_with: $starts_with:expr, ends_with: $ends_with:expr,
             relative_from: $relative_from:expr) => ({
                 let path1 = Path::new($path1);
                 let path2 = Path::new($path2);

                 let eq = path1 == path2;
                 assert!(eq == $eq, "{:?} == {:?}, expected {:?}, got {:?}",
                         $path1, $path2, $eq, eq);

                 let starts_with = path1.starts_with(path2);
                 assert!(starts_with == $starts_with,
                         "{:?}.starts_with({:?}), expected {:?}, got {:?}", $path1, $path2,
                         $starts_with, starts_with);

                 let ends_with = path1.ends_with(path2);
                 assert!(ends_with == $ends_with,
                         "{:?}.ends_with({:?}), expected {:?}, got {:?}", $path1, $path2,
                         $ends_with, ends_with);

                 let relative_from = path1.relative_from(path2).map(|p| p.to_str().unwrap());
                 let exp: Option<&str> = $relative_from;
                 assert!(relative_from == exp,
                         "{:?}.relative_from({:?}), expected {:?}, got {:?}", $path1, $path2,
                         exp, relative_from);
            });
        );

        tc!("", "",
            eq: true,
            starts_with: true,
            ends_with: true,
            relative_from: Some("")
            );

        tc!("foo", "",
            eq: false,
            starts_with: true,
            ends_with: true,
            relative_from: Some("foo")
            );

        tc!("", "foo",
            eq: false,
            starts_with: false,
            ends_with: false,
            relative_from: None
            );

        tc!("foo", "foo",
            eq: true,
            starts_with: true,
            ends_with: true,
            relative_from: Some("")
            );

        tc!("foo/", "foo",
            eq: true,
            starts_with: true,
            ends_with: true,
            relative_from: Some("")
            );

        tc!("foo/bar", "foo",
            eq: false,
            starts_with: true,
            ends_with: false,
            relative_from: Some("bar")
            );

        tc!("foo/bar/baz", "foo/bar",
            eq: false,
            starts_with: true,
            ends_with: false,
            relative_from: Some("baz")
            );

        tc!("foo/bar", "foo/bar/baz",
            eq: false,
            starts_with: false,
            ends_with: false,
            relative_from: None
            );

        tc!("./foo/bar/", ".",
            eq: false,
            starts_with: true,
            ends_with: false,
            relative_from: Some("foo/bar")
            );

        if cfg!(windows) {
            tc!(r"C:\src\rust\cargo-test\test\Cargo.toml",
                r"c:\src\rust\cargo-test\test",
                eq: false,
                starts_with: true,
                ends_with: false,
                relative_from: Some("Cargo.toml")
                );

            tc!(r"c:\foo", r"C:\foo",
                eq: true,
                starts_with: true,
                ends_with: true,
                relative_from: Some("")
                );
        }
    }
}
