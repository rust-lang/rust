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

use crate::borrow::{Borrow, Cow};
use crate::cmp;
use crate::error::Error;
use crate::fmt;
use crate::fs;
use crate::hash::{Hash, Hasher};
use crate::io;
use crate::iter::{self, FusedIterator};
use crate::ops::{self, Deref};
use crate::rc::Rc;
use crate::str::FromStr;
use crate::sync::Arc;

use crate::ffi::{OsStr, OsString};

use crate::sys::path::{is_sep_byte, is_verbatim_sep, parse_prefix, MAIN_SEP_STR};

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
// Windows Prefixes
////////////////////////////////////////////////////////////////////////////////

/// Windows path prefixes, e.g., `C:` or `\\server\share`.
///
/// Windows uses a variety of path prefix styles, including references to drive
/// volumes (like `C:`), network shared folders (like `\\server\share`), and
/// others. In addition, some path prefixes are "verbatim" (i.e., prefixed with
/// `\\?\`), in which case `/` is *not* treated as a separator and essentially
/// no normalization is performed.
///
/// # Examples
///
/// ```
/// use std::path::{Component, Path, Prefix};
/// use std::path::Prefix::*;
/// use std::ffi::OsStr;
///
/// fn get_path_prefix(s: &str) -> Prefix {
///     let path = Path::new(s);
///     match path.components().next().unwrap() {
///         Component::Prefix(prefix_component) => prefix_component.kind(),
///         _ => panic!(),
///     }
/// }
///
/// # if cfg!(windows) {
/// assert_eq!(Verbatim(OsStr::new("pictures")),
///            get_path_prefix(r"\\?\pictures\kittens"));
/// assert_eq!(VerbatimUNC(OsStr::new("server"), OsStr::new("share")),
///            get_path_prefix(r"\\?\UNC\server\share"));
/// assert_eq!(VerbatimDisk(b'C'), get_path_prefix(r"\\?\c:\"));
/// assert_eq!(DeviceNS(OsStr::new("BrainInterface")),
///            get_path_prefix(r"\\.\BrainInterface"));
/// assert_eq!(UNC(OsStr::new("server"), OsStr::new("share")),
///            get_path_prefix(r"\\server\share"));
/// assert_eq!(Disk(b'C'), get_path_prefix(r"C:\Users\Rust\Pictures\Ferris"));
/// # }
/// ```
#[derive(Copy, Clone, Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Prefix<'a> {
    /// Verbatim prefix, e.g., `\\?\cat_pics`.
    ///
    /// Verbatim prefixes consist of `\\?\` immediately followed by the given
    /// component.
    #[stable(feature = "rust1", since = "1.0.0")]
    Verbatim(#[stable(feature = "rust1", since = "1.0.0")] &'a OsStr),

    /// Verbatim prefix using Windows' _**U**niform **N**aming **C**onvention_,
    /// e.g., `\\?\UNC\server\share`.
    ///
    /// Verbatim UNC prefixes consist of `\\?\UNC\` immediately followed by the
    /// server's hostname and a share name.
    #[stable(feature = "rust1", since = "1.0.0")]
    VerbatimUNC(
        #[stable(feature = "rust1", since = "1.0.0")] &'a OsStr,
        #[stable(feature = "rust1", since = "1.0.0")] &'a OsStr,
    ),

    /// Verbatim disk prefix, e.g., `\\?\C:`.
    ///
    /// Verbatim disk prefixes consist of `\\?\` immediately followed by the
    /// drive letter and `:`.
    #[stable(feature = "rust1", since = "1.0.0")]
    VerbatimDisk(#[stable(feature = "rust1", since = "1.0.0")] u8),

    /// Device namespace prefix, e.g., `\\.\COM42`.
    ///
    /// Device namespace prefixes consist of `\\.\` immediately followed by the
    /// device name.
    #[stable(feature = "rust1", since = "1.0.0")]
    DeviceNS(#[stable(feature = "rust1", since = "1.0.0")] &'a OsStr),

    /// Prefix using Windows' _**U**niform **N**aming **C**onvention_, e.g.
    /// `\\server\share`.
    ///
    /// UNC prefixes consist of the server's hostname and a share name.
    #[stable(feature = "rust1", since = "1.0.0")]
    UNC(
        #[stable(feature = "rust1", since = "1.0.0")] &'a OsStr,
        #[stable(feature = "rust1", since = "1.0.0")] &'a OsStr,
    ),

    /// Prefix `C:` for the given disk drive.
    #[stable(feature = "rust1", since = "1.0.0")]
    Disk(#[stable(feature = "rust1", since = "1.0.0")] u8),
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
            VerbatimUNC(x, y) => {
                8 + os_str_len(x) + if os_str_len(y) > 0 { 1 + os_str_len(y) } else { 0 }
            }
            VerbatimDisk(_) => 6,
            UNC(x, y) => 2 + os_str_len(x) + if os_str_len(y) > 0 { 1 + os_str_len(y) } else { 0 },
            DeviceNS(x) => 4 + os_str_len(x),
            Disk(_) => 2,
        }
    }

    /// Determines if the prefix is verbatim, i.e., begins with `\\?\`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Prefix::*;
    /// use std::ffi::OsStr;
    ///
    /// assert!(Verbatim(OsStr::new("pictures")).is_verbatim());
    /// assert!(VerbatimUNC(OsStr::new("server"), OsStr::new("share")).is_verbatim());
    /// assert!(VerbatimDisk(b'C').is_verbatim());
    /// assert!(!DeviceNS(OsStr::new("BrainInterface")).is_verbatim());
    /// assert!(!UNC(OsStr::new("server"), OsStr::new("share")).is_verbatim());
    /// assert!(!Disk(b'C').is_verbatim());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_verbatim(&self) -> bool {
        use self::Prefix::*;
        matches!(*self, Verbatim(_) | VerbatimDisk(_) | VerbatimUNC(..))
    }

    #[inline]
    fn is_drive(&self) -> bool {
        matches!(*self, Prefix::Disk(_))
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
/// assert!(path::is_separator('/')); // '/' works for both Unix and Windows
/// assert!(!path::is_separator('❤'));
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn is_separator(c: char) -> bool {
    c.is_ascii() && is_sep_byte(c as u8)
}

/// The primary separator of path components for the current platform.
///
/// For example, `/` on Unix and `\` on Windows.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAIN_SEPARATOR: char = crate::sys::path::MAIN_SEP;

////////////////////////////////////////////////////////////////////////////////
// Misc helpers
////////////////////////////////////////////////////////////////////////////////

// Iterate through `iter` while it matches `prefix`; return `None` if `prefix`
// is not a prefix of `iter`, otherwise return `Some(iter_after_prefix)` giving
// `iter` after having exhausted `prefix`.
fn iter_after<'a, 'b, I, J>(mut iter: I, mut prefix: J) -> Option<I>
where
    I: Iterator<Item = Component<'a>> + Clone,
    J: Iterator<Item = Component<'b>>,
{
    loop {
        let mut iter_next = iter.clone();
        match (iter_next.next(), prefix.next()) {
            (Some(ref x), Some(ref y)) if x == y => (),
            (Some(_), Some(_)) => return None,
            (Some(_), None) => return Some(iter),
            (None, None) => return Some(iter),
            (None, Some(_)) => return None,
        }
        iter = iter_next;
    }
}

// See note at the top of this module to understand why these are used:
//
// These casts are safe as OsStr is internally a wrapper around [u8] on all
// platforms.
//
// Note that currently this relies on the special knowledge that libstd has;
// these types are single-element structs but are not marked repr(transparent)
// or repr(C) which would make these casts allowable outside std.
fn os_str_as_u8_slice(s: &OsStr) -> &[u8] {
    unsafe { &*(s as *const OsStr as *const [u8]) }
}
unsafe fn u8_slice_as_os_str(s: &[u8]) -> &OsStr {
    // SAFETY: see the comment of `os_str_as_u8_slice`
    unsafe { &*(s as *const [u8] as *const OsStr) }
}

// Detect scheme on Redox
fn has_redox_scheme(s: &[u8]) -> bool {
    cfg!(target_os = "redox") && s.contains(&b':')
}

////////////////////////////////////////////////////////////////////////////////
// Cross-platform, iterator-independent parsing
////////////////////////////////////////////////////////////////////////////////

/// Says whether the first byte after the prefix is a separator.
fn has_physical_root(s: &[u8], prefix: Option<Prefix<'_>>) -> bool {
    let path = if let Some(p) = prefix { &s[p.len()..] } else { s };
    !path.is_empty() && is_sep_byte(path[0])
}

// basic workhorse for splitting stem and extension
fn rsplit_file_at_dot(file: &OsStr) -> (Option<&OsStr>, Option<&OsStr>) {
    if os_str_as_u8_slice(file) == b".." {
        return (Some(file), None);
    }

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
        unsafe { (before.map(|s| u8_slice_as_os_str(s)), after.map(|s| u8_slice_as_os_str(s))) }
    }
}

fn split_file_at_dot(file: &OsStr) -> (&OsStr, Option<&OsStr>) {
    let slice = os_str_as_u8_slice(file);
    if slice == b".." {
        return (file, None);
    }

    // The unsafety here stems from converting between &OsStr and &[u8]
    // and back. This is safe to do because (1) we only look at ASCII
    // contents of the encoding and (2) new &OsStr values are produced
    // only from ASCII-bounded slices of existing &OsStr values.
    let i = match slice[1..].iter().position(|b| *b == b'.') {
        Some(i) => i + 1,
        None => return (file, None),
    };
    let before = &slice[..i];
    let after = &slice[i + 1..];
    unsafe { (u8_slice_as_os_str(before), Some(u8_slice_as_os_str(after))) }
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
    Prefix = 0,   // c:
    StartDir = 1, // / or . or nothing
    Body = 2,     // foo/bar/baz
    Done = 3,
}

/// A structure wrapping a Windows path prefix as well as its unparsed string
/// representation.
///
/// In addition to the parsed [`Prefix`] information returned by [`kind`],
/// `PrefixComponent` also holds the raw and unparsed [`OsStr`] slice,
/// returned by [`as_os_str`].
///
/// Instances of this `struct` can be obtained by matching against the
/// [`Prefix` variant] on [`Component`].
///
/// Does not occur on Unix.
///
/// # Examples
///
/// ```
/// # if cfg!(windows) {
/// use std::path::{Component, Path, Prefix};
/// use std::ffi::OsStr;
///
/// let path = Path::new(r"c:\you\later\");
/// match path.components().next().unwrap() {
///     Component::Prefix(prefix_component) => {
///         assert_eq!(Prefix::Disk(b'C'), prefix_component.kind());
///         assert_eq!(OsStr::new("c:"), prefix_component.as_os_str());
///     }
///     _ => unreachable!(),
/// }
/// # }
/// ```
///
/// [`as_os_str`]: PrefixComponent::as_os_str
/// [`kind`]: PrefixComponent::kind
/// [`Prefix` variant]: Component::Prefix
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone, Eq, Debug)]
pub struct PrefixComponent<'a> {
    /// The prefix as an unparsed `OsStr` slice.
    raw: &'a OsStr,

    /// The parsed prefix data.
    parsed: Prefix<'a>,
}

impl<'a> PrefixComponent<'a> {
    /// Returns the parsed prefix data.
    ///
    /// See [`Prefix`]'s documentation for more information on the different
    /// kinds of prefixes.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn kind(&self) -> Prefix<'a> {
        self.parsed
    }

    /// Returns the raw [`OsStr`] slice for this prefix.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn as_os_str(&self) -> &'a OsStr {
        self.raw
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::PartialEq for PrefixComponent<'a> {
    #[inline]
    fn eq(&self, other: &PrefixComponent<'a>) -> bool {
        cmp::PartialEq::eq(&self.parsed, &other.parsed)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::PartialOrd for PrefixComponent<'a> {
    #[inline]
    fn partial_cmp(&self, other: &PrefixComponent<'a>) -> Option<cmp::Ordering> {
        cmp::PartialOrd::partial_cmp(&self.parsed, &other.parsed)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Ord for PrefixComponent<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        cmp::Ord::cmp(&self.parsed, &other.parsed)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for PrefixComponent<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.parsed.hash(h);
    }
}

/// A single component of a path.
///
/// A `Component` roughly corresponds to a substring between path separators
/// (`/` or `\`).
///
/// This `enum` is created by iterating over [`Components`], which in turn is
/// created by the [`components`](Path::components) method on [`Path`].
///
/// # Examples
///
/// ```rust
/// use std::path::{Component, Path};
///
/// let path = Path::new("/tmp/foo/bar.txt");
/// let components = path.components().collect::<Vec<_>>();
/// assert_eq!(&components, &[
///     Component::RootDir,
///     Component::Normal("tmp".as_ref()),
///     Component::Normal("foo".as_ref()),
///     Component::Normal("bar.txt".as_ref()),
/// ]);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Component<'a> {
    /// A Windows path prefix, e.g., `C:` or `\\server\share`.
    ///
    /// There is a large variety of prefix types, see [`Prefix`]'s documentation
    /// for more.
    ///
    /// Does not occur on Unix.
    #[stable(feature = "rust1", since = "1.0.0")]
    Prefix(#[stable(feature = "rust1", since = "1.0.0")] PrefixComponent<'a>),

    /// The root directory component, appears after any prefix and before anything else.
    ///
    /// It represents a separator that designates that a path starts from root.
    #[stable(feature = "rust1", since = "1.0.0")]
    RootDir,

    /// A reference to the current directory, i.e., `.`.
    #[stable(feature = "rust1", since = "1.0.0")]
    CurDir,

    /// A reference to the parent directory, i.e., `..`.
    #[stable(feature = "rust1", since = "1.0.0")]
    ParentDir,

    /// A normal component, e.g., `a` and `b` in `a/b`.
    ///
    /// This variant is the most common one, it represents references to files
    /// or directories.
    #[stable(feature = "rust1", since = "1.0.0")]
    Normal(#[stable(feature = "rust1", since = "1.0.0")] &'a OsStr),
}

impl<'a> Component<'a> {
    /// Extracts the underlying [`OsStr`] slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("./tmp/foo/bar.txt");
    /// let components: Vec<_> = path.components().map(|comp| comp.as_os_str()).collect();
    /// assert_eq!(&components, &[".", "tmp", "foo", "bar.txt"]);
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
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
impl AsRef<OsStr> for Component<'_> {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        self.as_os_str()
    }
}

#[stable(feature = "path_component_asref", since = "1.25.0")]
impl AsRef<Path> for Component<'_> {
    #[inline]
    fn as_ref(&self) -> &Path {
        self.as_os_str().as_ref()
    }
}

/// An iterator over the [`Component`]s of a [`Path`].
///
/// This `struct` is created by the [`components`] method on [`Path`].
/// See its documentation for more.
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
///
/// [`components`]: Path::components
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Components<'a> {
    // The path left to parse components from
    path: &'a [u8],

    // The prefix as it was originally parsed, if any
    prefix: Option<Prefix<'a>>,

    // true if path *physically* has a root separator; for most Windows
    // prefixes, it may have a "logical" root separator for the purposes of
    // normalization, e.g.,  \\server\share == \\server\share\.
    has_physical_root: bool,

    // The iterator is double-ended, and these two states keep track of what has
    // been produced from either end
    front: State,
    back: State,
}

/// An iterator over the [`Component`]s of a [`Path`], as [`OsStr`] slices.
///
/// This `struct` is created by the [`iter`] method on [`Path`].
/// See its documentation for more.
///
/// [`iter`]: Path::iter
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a> {
    inner: Components<'a>,
}

#[stable(feature = "path_components_debug", since = "1.13.0")]
impl fmt::Debug for Components<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct DebugHelper<'a>(&'a Path);

        impl fmt::Debug for DebugHelper<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list().entries(self.0.components()).finish()
            }
        }

        f.debug_tuple("Components").field(&DebugHelper(self.as_path())).finish()
    }
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
        if self.front == State::Prefix { self.prefix_len() } else { 0 }
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
        if self.prefix_verbatim() { is_verbatim_sep(b) } else { is_sep_byte(b) }
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
        if comps.front == State::Body {
            comps.trim_left();
        }
        if comps.back == State::Body {
            comps.trim_right();
        }
        unsafe { Path::from_u8_slice(comps.path) }
    }

    /// Is the *original* path rooted?
    fn has_root(&self) -> bool {
        if self.has_physical_root {
            return true;
        }
        if let Some(p) = self.prefix {
            if p.has_implicit_root() {
                return true;
            }
        }
        false
    }

    /// Should the normalized path include a leading . ?
    fn include_cur_dir(&self) -> bool {
        if self.has_root() {
            return false;
        }
        let mut iter = self.path[self.prefix_len()..].iter();
        match (iter.next(), iter.next()) {
            (Some(&b'.'), None) => true,
            (Some(&b'.'), Some(&b)) => self.is_sep_byte(b),
            _ => false,
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
            _ => Some(Component::Normal(unsafe { u8_slice_as_os_str(comp) })),
        }
    }

    // parse a component from the left, saying how many bytes to consume to
    // remove the component
    fn parse_next_component(&self) -> (usize, Option<Component<'a>>) {
        debug_assert!(self.front == State::Body);
        let (extra, comp) = match self.path.iter().position(|b| self.is_sep_byte(*b)) {
            None => (0, self.path),
            Some(i) => (1, &self.path[..i]),
        };
        (comp.len() + extra, self.parse_single_component(comp))
    }

    // parse a component from the right, saying how many bytes to consume to
    // remove the component
    fn parse_next_component_back(&self) -> (usize, Option<Component<'a>>) {
        debug_assert!(self.back == State::Body);
        let start = self.len_before_body();
        let (extra, comp) = match self.path[start..].iter().rposition(|b| self.is_sep_byte(*b)) {
            None => (0, &self.path[start..]),
            Some(i) => (1, &self.path[start + i + 1..]),
        };
        (comp.len() + extra, self.parse_single_component(comp))
    }

    // trim away repeated separators (i.e., empty components) on the left
    fn trim_left(&mut self) {
        while !self.path.is_empty() {
            let (size, comp) = self.parse_next_component();
            if comp.is_some() {
                return;
            } else {
                self.path = &self.path[size..];
            }
        }
    }

    // trim away repeated separators (i.e., empty components) on the right
    fn trim_right(&mut self) {
        while self.path.len() > self.len_before_body() {
            let (size, comp) = self.parse_next_component_back();
            if comp.is_some() {
                return;
            } else {
                self.path = &self.path[..self.path.len() - size];
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for Components<'_> {
    #[inline]
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for Components<'_> {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        self.as_path().as_os_str()
    }
}

#[stable(feature = "path_iter_debug", since = "1.13.0")]
impl fmt::Debug for Iter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct DebugHelper<'a>(&'a Path);

        impl fmt::Debug for DebugHelper<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list().entries(self.0.iter()).finish()
            }
        }

        f.debug_tuple("Iter").field(&DebugHelper(self.as_path())).finish()
    }
}

impl<'a> Iter<'a> {
    /// Extracts a slice corresponding to the portion of the path remaining for iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let mut iter = Path::new("/tmp/foo/bar.txt").iter();
    /// iter.next();
    /// iter.next();
    ///
    /// assert_eq!(Path::new("foo/bar.txt"), iter.as_path());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn as_path(&self) -> &'a Path {
        self.inner.as_path()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for Iter<'_> {
    #[inline]
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for Iter<'_> {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        self.as_path().as_os_str()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Iter<'a> {
    type Item = &'a OsStr;

    #[inline]
    fn next(&mut self) -> Option<&'a OsStr> {
        self.inner.next().map(Component::as_os_str)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Iter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a OsStr> {
        self.inner.next_back().map(Component::as_os_str)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Iter<'_> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Components<'a> {
    type Item = Component<'a>;

    fn next(&mut self) -> Option<Component<'a>> {
        while !self.finished() {
            match self.front {
                State::Prefix if self.prefix_len() > 0 => {
                    self.front = State::StartDir;
                    debug_assert!(self.prefix_len() <= self.path.len());
                    let raw = &self.path[..self.prefix_len()];
                    self.path = &self.path[self.prefix_len()..];
                    return Some(Component::Prefix(PrefixComponent {
                        raw: unsafe { u8_slice_as_os_str(raw) },
                        parsed: self.prefix.unwrap(),
                    }));
                }
                State::Prefix => {
                    self.front = State::StartDir;
                }
                State::StartDir => {
                    self.front = State::Body;
                    if self.has_physical_root {
                        debug_assert!(!self.path.is_empty());
                        self.path = &self.path[1..];
                        return Some(Component::RootDir);
                    } else if let Some(p) = self.prefix {
                        if p.has_implicit_root() && !p.is_verbatim() {
                            return Some(Component::RootDir);
                        }
                    } else if self.include_cur_dir() {
                        debug_assert!(!self.path.is_empty());
                        self.path = &self.path[1..];
                        return Some(Component::CurDir);
                    }
                }
                State::Body if !self.path.is_empty() => {
                    let (size, comp) = self.parse_next_component();
                    self.path = &self.path[size..];
                    if comp.is_some() {
                        return comp;
                    }
                }
                State::Body => {
                    self.front = State::Done;
                }
                State::Done => unreachable!(),
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
                    self.path = &self.path[..self.path.len() - size];
                    if comp.is_some() {
                        return comp;
                    }
                }
                State::Body => {
                    self.back = State::StartDir;
                }
                State::StartDir => {
                    self.back = State::Prefix;
                    if self.has_physical_root {
                        self.path = &self.path[..self.path.len() - 1];
                        return Some(Component::RootDir);
                    } else if let Some(p) = self.prefix {
                        if p.has_implicit_root() && !p.is_verbatim() {
                            return Some(Component::RootDir);
                        }
                    } else if self.include_cur_dir() {
                        self.path = &self.path[..self.path.len() - 1];
                        return Some(Component::CurDir);
                    }
                }
                State::Prefix if self.prefix_len() > 0 => {
                    self.back = State::Done;
                    return Some(Component::Prefix(PrefixComponent {
                        raw: unsafe { u8_slice_as_os_str(self.path) },
                        parsed: self.prefix.unwrap(),
                    }));
                }
                State::Prefix => {
                    self.back = State::Done;
                    return None;
                }
                State::Done => unreachable!(),
            }
        }
        None
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Components<'_> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::PartialEq for Components<'a> {
    #[inline]
    fn eq(&self, other: &Components<'a>) -> bool {
        Iterator::eq(self.clone().rev(), other.clone().rev())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Eq for Components<'_> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> cmp::PartialOrd for Components<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Components<'a>) -> Option<cmp::Ordering> {
        Some(compare_components(self.clone(), other.clone()))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Ord for Components<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        compare_components(self.clone(), other.clone())
    }
}

fn compare_components(mut left: Components<'_>, mut right: Components<'_>) -> cmp::Ordering {
    // Fast path for long shared prefixes
    //
    // - compare raw bytes to find first mismatch
    // - backtrack to find separator before mismatch to avoid ambiguous parsings of '.' or '..' characters
    // - if found update state to only do a component-wise comparison on the remainder,
    //   otherwise do it on the full path
    //
    // The fast path isn't taken for paths with a PrefixComponent to avoid backtracking into
    // the middle of one
    if left.prefix.is_none() && right.prefix.is_none() && left.front == right.front {
        // this might benefit from a [u8]::first_mismatch simd implementation, if it existed
        let first_difference =
            match left.path.iter().zip(right.path.iter()).position(|(&a, &b)| a != b) {
                None if left.path.len() == right.path.len() => return cmp::Ordering::Equal,
                None => left.path.len().min(right.path.len()),
                Some(diff) => diff,
            };

        if let Some(previous_sep) =
            left.path[..first_difference].iter().rposition(|&b| left.is_sep_byte(b))
        {
            let mismatched_component_start = previous_sep + 1;
            left.path = &left.path[mismatched_component_start..];
            left.front = State::Body;
            right.path = &right.path[mismatched_component_start..];
            right.front = State::Body;
        }
    }

    Iterator::cmp(left, right)
}

/// An iterator over [`Path`] and its ancestors.
///
/// This `struct` is created by the [`ancestors`] method on [`Path`].
/// See its documentation for more.
///
/// # Examples
///
/// ```
/// use std::path::Path;
///
/// let path = Path::new("/foo/bar");
///
/// for ancestor in path.ancestors() {
///     println!("{}", ancestor.display());
/// }
/// ```
///
/// [`ancestors`]: Path::ancestors
#[derive(Copy, Clone, Debug)]
#[stable(feature = "path_ancestors", since = "1.28.0")]
pub struct Ancestors<'a> {
    next: Option<&'a Path>,
}

#[stable(feature = "path_ancestors", since = "1.28.0")]
impl<'a> Iterator for Ancestors<'a> {
    type Item = &'a Path;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next;
        self.next = next.and_then(Path::parent);
        next
    }
}

#[stable(feature = "path_ancestors", since = "1.28.0")]
impl FusedIterator for Ancestors<'_> {}

////////////////////////////////////////////////////////////////////////////////
// Basic types and traits
////////////////////////////////////////////////////////////////////////////////

/// An owned, mutable path (akin to [`String`]).
///
/// This type provides methods like [`push`] and [`set_extension`] that mutate
/// the path in place. It also implements [`Deref`] to [`Path`], meaning that
/// all methods on [`Path`] slices are available on `PathBuf` values as well.
///
/// [`push`]: PathBuf::push
/// [`set_extension`]: PathBuf::set_extension
///
/// More details about the overall approach can be found in
/// the [module documentation](self).
///
/// # Examples
///
/// You can use [`push`] to build up a `PathBuf` from
/// components:
///
/// ```
/// use std::path::PathBuf;
///
/// let mut path = PathBuf::new();
///
/// path.push(r"C:\");
/// path.push("windows");
/// path.push("system32");
///
/// path.set_extension("dll");
/// ```
///
/// However, [`push`] is best used for dynamic situations. This is a better way
/// to do this when you know all of the components ahead of time:
///
/// ```
/// use std::path::PathBuf;
///
/// let path: PathBuf = [r"C:\", "windows", "system32.dll"].iter().collect();
/// ```
///
/// We can still do better than this! Since these are all strings, we can use
/// `From::from`:
///
/// ```
/// use std::path::PathBuf;
///
/// let path = PathBuf::from(r"C:\windows\system32.dll");
/// ```
///
/// Which method works best depends on what kind of situation you're in.
#[cfg_attr(not(test), rustc_diagnostic_item = "PathBuf")]
#[stable(feature = "rust1", since = "1.0.0")]
// FIXME:
// `PathBuf::as_mut_vec` current implementation relies
// on `PathBuf` being layout-compatible with `Vec<u8>`.
// When attribute privacy is implemented, `PathBuf` should be annotated as `#[repr(transparent)]`.
// Anyway, `PathBuf` representation and layout are considered implementation detail, are
// not documented and must not be relied upon.
pub struct PathBuf {
    inner: OsString,
}

impl PathBuf {
    #[inline]
    fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        unsafe { &mut *(self as *mut PathBuf as *mut Vec<u8>) }
    }

    /// Allocates an empty `PathBuf`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let path = PathBuf::new();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub fn new() -> PathBuf {
        PathBuf { inner: OsString::new() }
    }

    /// Creates a new `PathBuf` with a given capacity used to create the
    /// internal [`OsString`]. See [`with_capacity`] defined on [`OsString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let mut path = PathBuf::with_capacity(10);
    /// let capacity = path.capacity();
    ///
    /// // This push is done without reallocating
    /// path.push(r"C:\");
    ///
    /// assert_eq!(capacity, path.capacity());
    /// ```
    ///
    /// [`with_capacity`]: OsString::with_capacity
    #[stable(feature = "path_buf_capacity", since = "1.44.0")]
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> PathBuf {
        PathBuf { inner: OsString::with_capacity(capacity) }
    }

    /// Coerces to a [`Path`] slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let p = PathBuf::from("/test");
    /// assert_eq!(Path::new("/test"), p.as_path());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn as_path(&self) -> &Path {
        self
    }

    /// Extends `self` with `path`.
    ///
    /// If `path` is absolute, it replaces the current path.
    ///
    /// On Windows:
    ///
    /// * if `path` has a root but no prefix (e.g., `\windows`), it
    ///   replaces everything except for the prefix (if any) of `self`.
    /// * if `path` has a prefix but no root, it replaces `self`.
    ///
    /// # Examples
    ///
    /// Pushing a relative path extends the existing path:
    ///
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let mut path = PathBuf::from("/tmp");
    /// path.push("file.bk");
    /// assert_eq!(path, PathBuf::from("/tmp/file.bk"));
    /// ```
    ///
    /// Pushing an absolute path replaces the existing path:
    ///
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let mut path = PathBuf::from("/tmp");
    /// path.push("/etc");
    /// assert_eq!(path, PathBuf::from("/etc"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push<P: AsRef<Path>>(&mut self, path: P) {
        self._push(path.as_ref())
    }

    fn _push(&mut self, path: &Path) {
        // in general, a separator is needed if the rightmost byte is not a separator
        let mut need_sep = self.as_mut_vec().last().map(|c| !is_sep_byte(*c)).unwrap_or(false);

        // in the special case of `C:` on Windows, do *not* add a separator
        let comps = self.components();

        if comps.prefix_len() > 0
            && comps.prefix_len() == comps.path.len()
            && comps.prefix.unwrap().is_drive()
        {
            need_sep = false
        }

        // absolute `path` replaces `self`
        if path.is_absolute() || path.prefix().is_some() {
            self.as_mut_vec().truncate(0);

        // verbatim paths need . and .. removed
        } else if comps.prefix_verbatim() {
            let mut buf: Vec<_> = comps.collect();
            for c in path.components() {
                match c {
                    Component::RootDir => {
                        buf.truncate(1);
                        buf.push(c);
                    }
                    Component::CurDir => (),
                    Component::ParentDir => {
                        if let Some(Component::Normal(_)) = buf.last() {
                            buf.pop();
                        }
                    }
                    _ => buf.push(c),
                }
            }

            let mut res = OsString::new();
            let mut need_sep = false;

            for c in buf {
                if need_sep && c != Component::RootDir {
                    res.push(MAIN_SEP_STR);
                }
                res.push(c.as_os_str());

                need_sep = match c {
                    Component::RootDir => false,
                    Component::Prefix(prefix) => {
                        !prefix.parsed.is_drive() && prefix.parsed.len() > 0
                    }
                    _ => true,
                }
            }

            self.inner = res;
            return;

        // `path` has a root but no prefix, e.g., `\windows` (Windows only)
        } else if path.has_root() {
            let prefix_len = self.components().prefix_remaining();
            self.as_mut_vec().truncate(prefix_len);

        // `path` is a pure relative path
        } else if need_sep {
            self.inner.push(MAIN_SEP_STR);
        }

        self.inner.push(path);
    }

    /// Truncates `self` to [`self.parent`].
    ///
    /// Returns `false` and does nothing if [`self.parent`] is [`None`].
    /// Otherwise, returns `true`.
    ///
    /// [`self.parent`]: Path::parent
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let mut p = PathBuf::from("/spirited/away.rs");
    ///
    /// p.pop();
    /// assert_eq!(Path::new("/spirited"), p);
    /// p.pop();
    /// assert_eq!(Path::new("/"), p);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop(&mut self) -> bool {
        match self.parent().map(|p| p.as_u8_slice().len()) {
            Some(len) => {
                self.as_mut_vec().truncate(len);
                true
            }
            None => false,
        }
    }

    /// Updates [`self.file_name`] to `file_name`.
    ///
    /// If [`self.file_name`] was [`None`], this is equivalent to pushing
    /// `file_name`.
    ///
    /// Otherwise it is equivalent to calling [`pop`] and then pushing
    /// `file_name`. The new path will be a sibling of the original path.
    /// (That is, it will have the same parent.)
    ///
    /// [`self.file_name`]: Path::file_name
    /// [`pop`]: PathBuf::pop
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

    /// Updates [`self.extension`] to `extension`.
    ///
    /// Returns `false` and does nothing if [`self.file_name`] is [`None`],
    /// returns `true` and updates the extension otherwise.
    ///
    /// If [`self.extension`] is [`None`], the extension is added; otherwise
    /// it is replaced.
    ///
    /// [`self.file_name`]: Path::file_name
    /// [`self.extension`]: Path::extension
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let mut p = PathBuf::from("/feel/the");
    ///
    /// p.set_extension("force");
    /// assert_eq!(Path::new("/feel/the.force"), p.as_path());
    ///
    /// p.set_extension("dark_side");
    /// assert_eq!(Path::new("/feel/the.dark_side"), p.as_path());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set_extension<S: AsRef<OsStr>>(&mut self, extension: S) -> bool {
        self._set_extension(extension.as_ref())
    }

    fn _set_extension(&mut self, extension: &OsStr) -> bool {
        let file_stem = match self.file_stem() {
            None => return false,
            Some(f) => os_str_as_u8_slice(f),
        };

        // truncate until right after the file stem
        let end_file_stem = file_stem[file_stem.len()..].as_ptr() as usize;
        let start = os_str_as_u8_slice(&self.inner).as_ptr() as usize;
        let v = self.as_mut_vec();
        v.truncate(end_file_stem.wrapping_sub(start));

        // add the new extension, if any
        let new = os_str_as_u8_slice(extension);
        if !new.is_empty() {
            v.reserve_exact(new.len() + 1);
            v.push(b'.');
            v.extend_from_slice(new);
        }

        true
    }

    /// Consumes the `PathBuf`, yielding its internal [`OsString`] storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let p = PathBuf::from("/the/head");
    /// let os_str = p.into_os_string();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub fn into_os_string(self) -> OsString {
        self.inner
    }

    /// Converts this `PathBuf` into a [boxed](Box) [`Path`].
    #[stable(feature = "into_boxed_path", since = "1.20.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub fn into_boxed_path(self) -> Box<Path> {
        let rw = Box::into_raw(self.inner.into_boxed_os_str()) as *mut Path;
        unsafe { Box::from_raw(rw) }
    }

    /// Invokes [`capacity`] on the underlying instance of [`OsString`].
    ///
    /// [`capacity`]: OsString::capacity
    #[stable(feature = "path_buf_capacity", since = "1.44.0")]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Invokes [`clear`] on the underlying instance of [`OsString`].
    ///
    /// [`clear`]: OsString::clear
    #[stable(feature = "path_buf_capacity", since = "1.44.0")]
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Invokes [`reserve`] on the underlying instance of [`OsString`].
    ///
    /// [`reserve`]: OsString::reserve
    #[stable(feature = "path_buf_capacity", since = "1.44.0")]
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    /// Invokes [`reserve_exact`] on the underlying instance of [`OsString`].
    ///
    /// [`reserve_exact`]: OsString::reserve_exact
    #[stable(feature = "path_buf_capacity", since = "1.44.0")]
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    /// Invokes [`shrink_to_fit`] on the underlying instance of [`OsString`].
    ///
    /// [`shrink_to_fit`]: OsString::shrink_to_fit
    #[stable(feature = "path_buf_capacity", since = "1.44.0")]
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    /// Invokes [`shrink_to`] on the underlying instance of [`OsString`].
    ///
    /// [`shrink_to`]: OsString::shrink_to
    #[stable(feature = "shrink_to", since = "1.56.0")]
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Clone for PathBuf {
    #[inline]
    fn clone(&self) -> Self {
        PathBuf { inner: self.inner.clone() }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner)
    }
}

#[stable(feature = "box_from_path", since = "1.17.0")]
impl From<&Path> for Box<Path> {
    /// Creates a boxed [`Path`] from a reference.
    ///
    /// This will allocate and clone `path` to it.
    fn from(path: &Path) -> Box<Path> {
        let boxed: Box<OsStr> = path.inner.into();
        let rw = Box::into_raw(boxed) as *mut Path;
        unsafe { Box::from_raw(rw) }
    }
}

#[stable(feature = "box_from_cow", since = "1.45.0")]
impl From<Cow<'_, Path>> for Box<Path> {
    /// Creates a boxed [`Path`] from a clone-on-write pointer.
    ///
    /// Converting from a `Cow::Owned` does not clone or allocate.
    #[inline]
    fn from(cow: Cow<'_, Path>) -> Box<Path> {
        match cow {
            Cow::Borrowed(path) => Box::from(path),
            Cow::Owned(path) => Box::from(path),
        }
    }
}

#[stable(feature = "path_buf_from_box", since = "1.18.0")]
impl From<Box<Path>> for PathBuf {
    /// Converts a `Box<Path>` into a `PathBuf`
    ///
    /// This conversion does not allocate or copy memory.
    #[inline]
    fn from(boxed: Box<Path>) -> PathBuf {
        boxed.into_path_buf()
    }
}

#[stable(feature = "box_from_path_buf", since = "1.20.0")]
impl From<PathBuf> for Box<Path> {
    /// Converts a `PathBuf` into a `Box<Path>`
    ///
    /// This conversion currently should not allocate memory,
    /// but this behavior is not guaranteed on all platforms or in all future versions.
    #[inline]
    fn from(p: PathBuf) -> Box<Path> {
        p.into_boxed_path()
    }
}

#[stable(feature = "more_box_slice_clone", since = "1.29.0")]
impl Clone for Box<Path> {
    #[inline]
    fn clone(&self) -> Self {
        self.to_path_buf().into_boxed_path()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + AsRef<OsStr>> From<&T> for PathBuf {
    /// Converts a borrowed `OsStr` to a `PathBuf`.
    ///
    /// Allocates a [`PathBuf`] and copies the data into it.
    #[inline]
    fn from(s: &T) -> PathBuf {
        PathBuf::from(s.as_ref().to_os_string())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<OsString> for PathBuf {
    /// Converts an [`OsString`] into a [`PathBuf`]
    ///
    /// This conversion does not allocate or copy memory.
    #[inline]
    fn from(s: OsString) -> PathBuf {
        PathBuf { inner: s }
    }
}

#[stable(feature = "from_path_buf_for_os_string", since = "1.14.0")]
impl From<PathBuf> for OsString {
    /// Converts a [`PathBuf`] into an [`OsString`]
    ///
    /// This conversion does not allocate or copy memory.
    #[inline]
    fn from(path_buf: PathBuf) -> OsString {
        path_buf.inner
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<String> for PathBuf {
    /// Converts a [`String`] into a [`PathBuf`]
    ///
    /// This conversion does not allocate or copy memory.
    #[inline]
    fn from(s: String) -> PathBuf {
        PathBuf::from(OsString::from(s))
    }
}

#[stable(feature = "path_from_str", since = "1.32.0")]
impl FromStr for PathBuf {
    type Err = core::convert::Infallible;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(PathBuf::from(s))
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
        iter.into_iter().for_each(move |p| self.push(p.as_ref()));
    }

    #[inline]
    fn extend_one(&mut self, p: P) {
        self.push(p.as_ref());
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for PathBuf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, formatter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Deref for PathBuf {
    type Target = Path;
    #[inline]
    fn deref(&self) -> &Path {
        Path::new(&self.inner)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Borrow<Path> for PathBuf {
    #[inline]
    fn borrow(&self) -> &Path {
        self.deref()
    }
}

#[stable(feature = "default_for_pathbuf", since = "1.17.0")]
impl Default for PathBuf {
    #[inline]
    fn default() -> Self {
        PathBuf::new()
    }
}

#[stable(feature = "cow_from_path", since = "1.6.0")]
impl<'a> From<&'a Path> for Cow<'a, Path> {
    /// Creates a clone-on-write pointer from a reference to
    /// [`Path`].
    ///
    /// This conversion does not clone or allocate.
    #[inline]
    fn from(s: &'a Path) -> Cow<'a, Path> {
        Cow::Borrowed(s)
    }
}

#[stable(feature = "cow_from_path", since = "1.6.0")]
impl<'a> From<PathBuf> for Cow<'a, Path> {
    /// Creates a clone-on-write pointer from an owned
    /// instance of [`PathBuf`].
    ///
    /// This conversion does not clone or allocate.
    #[inline]
    fn from(s: PathBuf) -> Cow<'a, Path> {
        Cow::Owned(s)
    }
}

#[stable(feature = "cow_from_pathbuf_ref", since = "1.28.0")]
impl<'a> From<&'a PathBuf> for Cow<'a, Path> {
    /// Creates a clone-on-write pointer from a reference to
    /// [`PathBuf`].
    ///
    /// This conversion does not clone or allocate.
    #[inline]
    fn from(p: &'a PathBuf) -> Cow<'a, Path> {
        Cow::Borrowed(p.as_path())
    }
}

#[stable(feature = "pathbuf_from_cow_path", since = "1.28.0")]
impl<'a> From<Cow<'a, Path>> for PathBuf {
    /// Converts a clone-on-write pointer to an owned path.
    ///
    /// Converting from a `Cow::Owned` does not clone or allocate.
    #[inline]
    fn from(p: Cow<'a, Path>) -> Self {
        p.into_owned()
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<PathBuf> for Arc<Path> {
    /// Converts a [`PathBuf`] into an [`Arc`] by moving the [`PathBuf`] data into a new [`Arc`] buffer.
    #[inline]
    fn from(s: PathBuf) -> Arc<Path> {
        let arc: Arc<OsStr> = Arc::from(s.into_os_string());
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Path) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<&Path> for Arc<Path> {
    /// Converts a [`Path`] into an [`Arc`] by copying the [`Path`] data into a new [`Arc`] buffer.
    #[inline]
    fn from(s: &Path) -> Arc<Path> {
        let arc: Arc<OsStr> = Arc::from(s.as_os_str());
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Path) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<PathBuf> for Rc<Path> {
    /// Converts a [`PathBuf`] into an [`Rc`] by moving the [`PathBuf`] data into a new `Rc` buffer.
    #[inline]
    fn from(s: PathBuf) -> Rc<Path> {
        let rc: Rc<OsStr> = Rc::from(s.into_os_string());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const Path) }
    }
}

#[stable(feature = "shared_from_slice2", since = "1.24.0")]
impl From<&Path> for Rc<Path> {
    /// Converts a [`Path`] into an [`Rc`] by copying the [`Path`] data into a new `Rc` buffer.
    #[inline]
    fn from(s: &Path) -> Rc<Path> {
        let rc: Rc<OsStr> = Rc::from(s.as_os_str());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const Path) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToOwned for Path {
    type Owned = PathBuf;
    #[inline]
    fn to_owned(&self) -> PathBuf {
        self.to_path_buf()
    }
    #[inline]
    fn clone_into(&self, target: &mut PathBuf) {
        self.inner.clone_into(&mut target.inner);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialEq for PathBuf {
    #[inline]
    fn eq(&self, other: &PathBuf) -> bool {
        self.components() == other.components()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for PathBuf {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.as_path().hash(h)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Eq for PathBuf {}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialOrd for PathBuf {
    #[inline]
    fn partial_cmp(&self, other: &PathBuf) -> Option<cmp::Ordering> {
        Some(compare_components(self.components(), other.components()))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Ord for PathBuf {
    #[inline]
    fn cmp(&self, other: &PathBuf) -> cmp::Ordering {
        compare_components(self.components(), other.components())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for PathBuf {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        &self.inner[..]
    }
}

/// A slice of a path (akin to [`str`]).
///
/// This type supports a number of operations for inspecting a path, including
/// breaking the path into its components (separated by `/` on Unix and by either
/// `/` or `\` on Windows), extracting the file name, determining whether the path
/// is absolute, and so on.
///
/// This is an *unsized* type, meaning that it must always be used behind a
/// pointer like `&` or [`Box`]. For an owned version of this type,
/// see [`PathBuf`].
///
/// More details about the overall approach can be found in
/// the [module documentation](self).
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use std::ffi::OsStr;
///
/// // Note: this example does work on Windows
/// let path = Path::new("./foo/bar.txt");
///
/// let parent = path.parent();
/// assert_eq!(parent, Some(Path::new("./foo")));
///
/// let file_stem = path.file_stem();
/// assert_eq!(file_stem, Some(OsStr::new("bar")));
///
/// let extension = path.extension();
/// assert_eq!(extension, Some(OsStr::new("txt")));
/// ```
#[cfg_attr(not(test), rustc_diagnostic_item = "Path")]
#[stable(feature = "rust1", since = "1.0.0")]
// FIXME:
// `Path::new` current implementation relies
// on `Path` being layout-compatible with `OsStr`.
// When attribute privacy is implemented, `Path` should be annotated as `#[repr(transparent)]`.
// Anyway, `Path` representation and layout are considered implementation detail, are
// not documented and must not be relied upon.
pub struct Path {
    inner: OsStr,
}

/// An error returned from [`Path::strip_prefix`] if the prefix was not found.
///
/// This `struct` is created by the [`strip_prefix`] method on [`Path`].
/// See its documentation for more.
///
/// [`strip_prefix`]: Path::strip_prefix
#[derive(Debug, Clone, PartialEq, Eq)]
#[stable(since = "1.7.0", feature = "strip_prefix")]
pub struct StripPrefixError(());

impl Path {
    // The following (private!) function allows construction of a path from a u8
    // slice, which is only safe when it is known to follow the OsStr encoding.
    unsafe fn from_u8_slice(s: &[u8]) -> &Path {
        unsafe { Path::new(u8_slice_as_os_str(s)) }
    }
    // The following (private!) function reveals the byte encoding used for OsStr.
    fn as_u8_slice(&self) -> &[u8] {
        os_str_as_u8_slice(&self.inner)
    }

    /// Directly wraps a string slice as a `Path` slice.
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
        unsafe { &*(s.as_ref() as *const OsStr as *const Path) }
    }

    /// Yields the underlying [`OsStr`] slice.
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
    #[inline]
    pub fn as_os_str(&self) -> &OsStr {
        &self.inner
    }

    /// Yields a [`&str`] slice if the `Path` is valid unicode.
    ///
    /// This conversion may entail doing a check for UTF-8 validity.
    /// Note that validation is performed because non-UTF-8 strings are
    /// perfectly valid for some OS.
    ///
    /// [`&str`]: str
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("foo.txt");
    /// assert_eq!(path.to_str(), Some("foo.txt"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_str(&self) -> Option<&str> {
        self.inner.to_str()
    }

    /// Converts a `Path` to a [`Cow<str>`].
    ///
    /// Any non-Unicode sequences are replaced with
    /// [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD].
    ///
    /// [U+FFFD]: super::char::REPLACEMENT_CHARACTER
    ///
    /// # Examples
    ///
    /// Calling `to_string_lossy` on a `Path` with valid unicode:
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path = Path::new("foo.txt");
    /// assert_eq!(path.to_string_lossy(), "foo.txt");
    /// ```
    ///
    /// Had `path` contained invalid unicode, the `to_string_lossy` call might
    /// have returned `"fo�.txt"`.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.inner.to_string_lossy()
    }

    /// Converts a `Path` to an owned [`PathBuf`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let path_buf = Path::new("foo.txt").to_path_buf();
    /// assert_eq!(path_buf, std::path::PathBuf::from("foo.txt"));
    /// ```
    #[rustc_conversion_suggestion]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_path_buf(&self) -> PathBuf {
        PathBuf::from(self.inner.to_os_string())
    }

    /// Returns `true` if the `Path` is absolute, i.e., if it is independent of
    /// the current directory.
    ///
    /// * On Unix, a path is absolute if it starts with the root, so
    /// `is_absolute` and [`has_root`] are equivalent.
    ///
    /// * On Windows, a path is absolute if it has a prefix and starts with the
    /// root: `c:\windows` is absolute, while `c:temp` and `\temp` are not.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// assert!(!Path::new("foo.txt").is_absolute());
    /// ```
    ///
    /// [`has_root`]: Path::has_root
    #[stable(feature = "rust1", since = "1.0.0")]
    #[allow(deprecated)]
    pub fn is_absolute(&self) -> bool {
        if cfg!(target_os = "redox") {
            // FIXME: Allow Redox prefixes
            self.has_root() || has_redox_scheme(self.as_u8_slice())
        } else {
            self.has_root() && (cfg!(any(unix, target_os = "wasi")) || self.prefix().is_some())
        }
    }

    /// Returns `true` if the `Path` is relative, i.e., not absolute.
    ///
    /// See [`is_absolute`]'s documentation for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// assert!(Path::new("foo.txt").is_relative());
    /// ```
    ///
    /// [`is_absolute`]: Path::is_absolute
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_relative(&self) -> bool {
        !self.is_absolute()
    }

    fn prefix(&self) -> Option<Prefix<'_>> {
        self.components().prefix
    }

    /// Returns `true` if the `Path` has a root.
    ///
    /// * On Unix, a path has a root if it begins with `/`.
    ///
    /// * On Windows, a path has a root if it:
    ///     * has no prefix and begins with a separator, e.g., `\windows`
    ///     * has a prefix followed by a separator, e.g., `c:\windows` but not `c:windows`
    ///     * has any non-disk prefix, e.g., `\\server\share`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// assert!(Path::new("/etc/passwd").has_root());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn has_root(&self) -> bool {
        self.components().has_root()
    }

    /// Returns the `Path` without its final component, if there is one.
    ///
    /// Returns [`None`] if the path terminates in a root or prefix.
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
            Component::Normal(_) | Component::CurDir | Component::ParentDir => {
                Some(comps.as_path())
            }
            _ => None,
        })
    }

    /// Produces an iterator over `Path` and its ancestors.
    ///
    /// The iterator will yield the `Path` that is returned if the [`parent`] method is used zero
    /// or more times. That means, the iterator will yield `&self`, `&self.parent().unwrap()`,
    /// `&self.parent().unwrap().parent().unwrap()` and so on. If the [`parent`] method returns
    /// [`None`], the iterator will do likewise. The iterator will always yield at least one value,
    /// namely `&self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// let mut ancestors = Path::new("/foo/bar").ancestors();
    /// assert_eq!(ancestors.next(), Some(Path::new("/foo/bar")));
    /// assert_eq!(ancestors.next(), Some(Path::new("/foo")));
    /// assert_eq!(ancestors.next(), Some(Path::new("/")));
    /// assert_eq!(ancestors.next(), None);
    ///
    /// let mut ancestors = Path::new("../foo/bar").ancestors();
    /// assert_eq!(ancestors.next(), Some(Path::new("../foo/bar")));
    /// assert_eq!(ancestors.next(), Some(Path::new("../foo")));
    /// assert_eq!(ancestors.next(), Some(Path::new("..")));
    /// assert_eq!(ancestors.next(), Some(Path::new("")));
    /// assert_eq!(ancestors.next(), None);
    /// ```
    ///
    /// [`parent`]: Path::parent
    #[stable(feature = "path_ancestors", since = "1.28.0")]
    #[inline]
    pub fn ancestors(&self) -> Ancestors<'_> {
        Ancestors { next: Some(&self) }
    }

    /// Returns the final component of the `Path`, if there is one.
    ///
    /// If the path is a normal file, this is the file name. If it's the path of a directory, this
    /// is the directory name.
    ///
    /// Returns [`None`] if the path terminates in `..`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use std::ffi::OsStr;
    ///
    /// assert_eq!(Some(OsStr::new("bin")), Path::new("/usr/bin/").file_name());
    /// assert_eq!(Some(OsStr::new("foo.txt")), Path::new("tmp/foo.txt").file_name());
    /// assert_eq!(Some(OsStr::new("foo.txt")), Path::new("foo.txt/.").file_name());
    /// assert_eq!(Some(OsStr::new("foo.txt")), Path::new("foo.txt/.//").file_name());
    /// assert_eq!(None, Path::new("foo.txt/..").file_name());
    /// assert_eq!(None, Path::new("/").file_name());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn file_name(&self) -> Option<&OsStr> {
        self.components().next_back().and_then(|p| match p {
            Component::Normal(p) => Some(p),
            _ => None,
        })
    }

    /// Returns a path that, when joined onto `base`, yields `self`.
    ///
    /// # Errors
    ///
    /// If `base` is not a prefix of `self` (i.e., [`starts_with`]
    /// returns `false`), returns [`Err`].
    ///
    /// [`starts_with`]: Path::starts_with
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let path = Path::new("/test/haha/foo.txt");
    ///
    /// assert_eq!(path.strip_prefix("/"), Ok(Path::new("test/haha/foo.txt")));
    /// assert_eq!(path.strip_prefix("/test"), Ok(Path::new("haha/foo.txt")));
    /// assert_eq!(path.strip_prefix("/test/"), Ok(Path::new("haha/foo.txt")));
    /// assert_eq!(path.strip_prefix("/test/haha/foo.txt"), Ok(Path::new("")));
    /// assert_eq!(path.strip_prefix("/test/haha/foo.txt/"), Ok(Path::new("")));
    ///
    /// assert!(path.strip_prefix("test").is_err());
    /// assert!(path.strip_prefix("/haha").is_err());
    ///
    /// let prefix = PathBuf::from("/test/");
    /// assert_eq!(path.strip_prefix(prefix), Ok(Path::new("haha/foo.txt")));
    /// ```
    #[stable(since = "1.7.0", feature = "path_strip_prefix")]
    pub fn strip_prefix<P>(&self, base: P) -> Result<&Path, StripPrefixError>
    where
        P: AsRef<Path>,
    {
        self._strip_prefix(base.as_ref())
    }

    fn _strip_prefix(&self, base: &Path) -> Result<&Path, StripPrefixError> {
        iter_after(self.components(), base.components())
            .map(|c| c.as_path())
            .ok_or(StripPrefixError(()))
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
    /// assert!(path.starts_with("/etc/"));
    /// assert!(path.starts_with("/etc/passwd"));
    /// assert!(path.starts_with("/etc/passwd/")); // extra slash is okay
    /// assert!(path.starts_with("/etc/passwd///")); // multiple extra slashes are okay
    ///
    /// assert!(!path.starts_with("/e"));
    /// assert!(!path.starts_with("/etc/passwd.txt"));
    ///
    /// assert!(!Path::new("/etc/foo.rs").starts_with("/etc/foo"));
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
    /// let path = Path::new("/etc/resolv.conf");
    ///
    /// assert!(path.ends_with("resolv.conf"));
    /// assert!(path.ends_with("etc/resolv.conf"));
    /// assert!(path.ends_with("/etc/resolv.conf"));
    ///
    /// assert!(!path.ends_with("/resolv.conf"));
    /// assert!(!path.ends_with("conf")); // use .extension() instead
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ends_with<P: AsRef<Path>>(&self, child: P) -> bool {
        self._ends_with(child.as_ref())
    }

    fn _ends_with(&self, child: &Path) -> bool {
        iter_after(self.components().rev(), child.components().rev()).is_some()
    }

    /// Extracts the stem (non-extension) portion of [`self.file_name`].
    ///
    /// [`self.file_name`]: Path::file_name
    ///
    /// The stem is:
    ///
    /// * [`None`], if there is no file name;
    /// * The entire file name if there is no embedded `.`;
    /// * The entire file name if the file name begins with `.` and has no other `.`s within;
    /// * Otherwise, the portion of the file name before the final `.`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// assert_eq!("foo", Path::new("foo.rs").file_stem().unwrap());
    /// assert_eq!("foo.tar", Path::new("foo.tar.gz").file_stem().unwrap());
    /// ```
    ///
    /// # See Also
    /// This method is similar to [`Path::file_prefix`], which extracts the portion of the file name
    /// before the *first* `.`
    ///
    /// [`Path::file_prefix`]: Path::file_prefix
    ///
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn file_stem(&self) -> Option<&OsStr> {
        self.file_name().map(rsplit_file_at_dot).and_then(|(before, after)| before.or(after))
    }

    /// Extracts the prefix of [`self.file_name`].
    ///
    /// The prefix is:
    ///
    /// * [`None`], if there is no file name;
    /// * The entire file name if there is no embedded `.`;
    /// * The portion of the file name before the first non-beginning `.`;
    /// * The entire file name if the file name begins with `.` and has no other `.`s within;
    /// * The portion of the file name before the second `.` if the file name begins with `.`
    ///
    /// [`self.file_name`]: Path::file_name
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(path_file_prefix)]
    /// use std::path::Path;
    ///
    /// assert_eq!("foo", Path::new("foo.rs").file_prefix().unwrap());
    /// assert_eq!("foo", Path::new("foo.tar.gz").file_prefix().unwrap());
    /// ```
    ///
    /// # See Also
    /// This method is similar to [`Path::file_stem`], which extracts the portion of the file name
    /// before the *last* `.`
    ///
    /// [`Path::file_stem`]: Path::file_stem
    ///
    #[unstable(feature = "path_file_prefix", issue = "86319")]
    pub fn file_prefix(&self) -> Option<&OsStr> {
        self.file_name().map(split_file_at_dot).and_then(|(before, _after)| Some(before))
    }

    /// Extracts the extension of [`self.file_name`], if possible.
    ///
    /// The extension is:
    ///
    /// * [`None`], if there is no file name;
    /// * [`None`], if there is no embedded `.`;
    /// * [`None`], if the file name begins with `.` and has no other `.`s within;
    /// * Otherwise, the portion of the file name after the final `.`
    ///
    /// [`self.file_name`]: Path::file_name
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// assert_eq!("rs", Path::new("foo.rs").extension().unwrap());
    /// assert_eq!("gz", Path::new("foo.tar.gz").extension().unwrap());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn extension(&self) -> Option<&OsStr> {
        self.file_name().map(rsplit_file_at_dot).and_then(|(before, after)| before.and(after))
    }

    /// Creates an owned [`PathBuf`] with `path` adjoined to `self`.
    ///
    /// See [`PathBuf::push`] for more details on what it means to adjoin a path.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// assert_eq!(Path::new("/etc").join("passwd"), PathBuf::from("/etc/passwd"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    pub fn join<P: AsRef<Path>>(&self, path: P) -> PathBuf {
        self._join(path.as_ref())
    }

    fn _join(&self, path: &Path) -> PathBuf {
        let mut buf = self.to_path_buf();
        buf.push(path);
        buf
    }

    /// Creates an owned [`PathBuf`] like `self` but with the given file name.
    ///
    /// See [`PathBuf::set_file_name`] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let path = Path::new("/tmp/foo.txt");
    /// assert_eq!(path.with_file_name("bar.txt"), PathBuf::from("/tmp/bar.txt"));
    ///
    /// let path = Path::new("/tmp");
    /// assert_eq!(path.with_file_name("var"), PathBuf::from("/var"));
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

    /// Creates an owned [`PathBuf`] like `self` but with the given extension.
    ///
    /// See [`PathBuf::set_extension`] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let path = Path::new("foo.rs");
    /// assert_eq!(path.with_extension("txt"), PathBuf::from("foo.txt"));
    ///
    /// let path = Path::new("foo.tar.gz");
    /// assert_eq!(path.with_extension(""), PathBuf::from("foo.tar"));
    /// assert_eq!(path.with_extension("xz"), PathBuf::from("foo.tar.xz"));
    /// assert_eq!(path.with_extension("").with_extension("txt"), PathBuf::from("foo.txt"));
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

    /// Produces an iterator over the [`Component`]s of the path.
    ///
    /// When parsing the path, there is a small amount of normalization:
    ///
    /// * Repeated separators are ignored, so `a/b` and `a//b` both have
    ///   `a` and `b` as components.
    ///
    /// * Occurrences of `.` are normalized away, except if they are at the
    ///   beginning of the path. For example, `a/./b`, `a/b/`, `a/b/.` and
    ///   `a/b` all have `a` and `b` as components, but `./a/b` starts with
    ///   an additional [`CurDir`] component.
    ///
    /// * A trailing slash is normalized away, `/a/b` and `/a/b/` are equivalent.
    ///
    /// Note that no other normalization takes place; in particular, `a/c`
    /// and `a/b/../c` are distinct, to account for the possibility that `b`
    /// is a symbolic link (so its parent isn't `a`).
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
    ///
    /// [`CurDir`]: Component::CurDir
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn components(&self) -> Components<'_> {
        let prefix = parse_prefix(self.as_os_str());
        Components {
            path: self.as_u8_slice(),
            prefix,
            has_physical_root: has_physical_root(self.as_u8_slice(), prefix)
                || has_redox_scheme(self.as_u8_slice()),
            front: State::Prefix,
            back: State::Body,
        }
    }

    /// Produces an iterator over the path's components viewed as [`OsStr`]
    /// slices.
    ///
    /// For more information about the particulars of how the path is separated
    /// into components, see [`components`].
    ///
    /// [`components`]: Path::components
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
    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        Iter { inner: self.components() }
    }

    /// Returns an object that implements [`Display`] for safely printing paths
    /// that may contain non-Unicode data. This may perform lossy conversion,
    /// depending on the platform.  If you would like an implementation which
    /// escapes the path please use [`Debug`] instead.
    ///
    /// [`Display`]: fmt::Display
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
    #[must_use = "this does not display the path, \
                  it returns an object that can be displayed"]
    #[inline]
    pub fn display(&self) -> Display<'_> {
        Display { path: self }
    }

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
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[inline]
    pub fn read_dir(&self) -> io::Result<fs::ReadDir> {
        fs::read_dir(self)
    }

    /// Returns `true` if the path points at an existing entity.
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
    /// check errors, call [`fs::metadata`].
    #[stable(feature = "path_ext", since = "1.5.0")]
    #[inline]
    pub fn exists(&self) -> bool {
        fs::metadata(self).is_ok()
    }

    /// Returns `Ok(true)` if the path points at an existing entity.
    ///
    /// This function will traverse symbolic links to query information about the
    /// destination file. In case of broken symbolic links this will return `Ok(false)`.
    ///
    /// As opposed to the `exists()` method, this one doesn't silently ignore errors
    /// unrelated to the path not existing. (E.g. it will return `Err(_)` in case of permission
    /// denied on some of the parent directories.)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(path_try_exists)]
    ///
    /// use std::path::Path;
    /// assert!(!Path::new("does_not_exist.txt").try_exists().expect("Can't check existence of file does_not_exist.txt"));
    /// assert!(Path::new("/root/secret_file.txt").try_exists().is_err());
    /// ```
    // FIXME: stabilization should modify documentation of `exists()` to recommend this method
    // instead.
    #[unstable(feature = "path_try_exists", issue = "83186")]
    #[inline]
    pub fn try_exists(&self) -> io::Result<bool> {
        fs::try_exists(self)
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
    #[stable(feature = "path_ext", since = "1.5.0")]
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
    #[stable(feature = "path_ext", since = "1.5.0")]
    pub fn is_dir(&self) -> bool {
        fs::metadata(self).map(|m| m.is_dir()).unwrap_or(false)
    }

    /// Returns true if the path exists on disk and is pointing at a symbolic link.
    ///
    /// This function will not traverse symbolic links.
    /// In case of a broken symbolic link this will also return true.
    ///
    /// If you cannot access the directory containing the file, e.g., because of a
    /// permission error, this will return false.
    ///
    /// # Examples
    ///
    #[cfg_attr(unix, doc = "```no_run")]
    #[cfg_attr(not(unix), doc = "```ignore")]
    /// #![feature(is_symlink)]
    /// use std::path::Path;
    /// use std::os::unix::fs::symlink;
    ///
    /// let link_path = Path::new("link");
    /// symlink("/origin_does_not_exists/", link_path).unwrap();
    /// assert_eq!(link_path.is_symlink(), true);
    /// assert_eq!(link_path.exists(), false);
    /// ```
    #[unstable(feature = "is_symlink", issue = "85748")]
    pub fn is_symlink(&self) -> bool {
        fs::symlink_metadata(self).map(|m| m.is_symlink()).unwrap_or(false)
    }

    /// Converts a [`Box<Path>`](Box) into a [`PathBuf`] without copying or
    /// allocating.
    #[stable(feature = "into_boxed_path", since = "1.20.0")]
    pub fn into_path_buf(self: Box<Path>) -> PathBuf {
        let rw = Box::into_raw(self) as *mut OsStr;
        let inner = unsafe { Box::from_raw(rw) };
        PathBuf { inner: OsString::from(inner) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for Path {
    #[inline]
    fn as_ref(&self) -> &OsStr {
        &self.inner
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Path {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

/// Helper struct for safely printing paths with [`format!`] and `{}`.
///
/// A [`Path`] might contain non-Unicode data. This `struct` implements the
/// [`Display`] trait in a way that mitigates that. It is created by the
/// [`display`](Path::display) method on [`Path`]. This may perform lossy
/// conversion, depending on the platform. If you would like an implementation
/// which escapes the path please use [`Debug`] instead.
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
///
/// [`Display`]: fmt::Display
/// [`format!`]: crate::format
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Display<'a> {
    path: &'a Path,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.path, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.path.inner.display(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialEq for Path {
    #[inline]
    fn eq(&self, other: &Path) -> bool {
        self.components() == other.components()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for Path {
    fn hash<H: Hasher>(&self, h: &mut H) {
        for component in self.components() {
            component.hash(h);
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Eq for Path {}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialOrd for Path {
    #[inline]
    fn partial_cmp(&self, other: &Path) -> Option<cmp::Ordering> {
        Some(compare_components(self.components(), other.components()))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Ord for Path {
    #[inline]
    fn cmp(&self, other: &Path) -> cmp::Ordering {
        compare_components(self.components(), other.components())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for Path {
    #[inline]
    fn as_ref(&self) -> &Path {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for OsStr {
    #[inline]
    fn as_ref(&self) -> &Path {
        Path::new(self)
    }
}

#[stable(feature = "cow_os_str_as_ref_path", since = "1.8.0")]
impl AsRef<Path> for Cow<'_, OsStr> {
    #[inline]
    fn as_ref(&self) -> &Path {
        Path::new(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for OsString {
    #[inline]
    fn as_ref(&self) -> &Path {
        Path::new(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for str {
    #[inline]
    fn as_ref(&self) -> &Path {
        Path::new(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for String {
    #[inline]
    fn as_ref(&self) -> &Path {
        Path::new(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for PathBuf {
    #[inline]
    fn as_ref(&self) -> &Path {
        self
    }
}

#[stable(feature = "path_into_iter", since = "1.6.0")]
impl<'a> IntoIterator for &'a PathBuf {
    type Item = &'a OsStr;
    type IntoIter = Iter<'a>;
    #[inline]
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

#[stable(feature = "path_into_iter", since = "1.6.0")]
impl<'a> IntoIterator for &'a Path {
    type Item = &'a OsStr;
    type IntoIter = Iter<'a>;
    #[inline]
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

macro_rules! impl_cmp {
    ($lhs:ty, $rhs: ty) => {
        #[stable(feature = "partialeq_path", since = "1.6.0")]
        impl<'a, 'b> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                <Path as PartialEq>::eq(self, other)
            }
        }

        #[stable(feature = "partialeq_path", since = "1.6.0")]
        impl<'a, 'b> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                <Path as PartialEq>::eq(self, other)
            }
        }

        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<'a, 'b> PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<cmp::Ordering> {
                <Path as PartialOrd>::partial_cmp(self, other)
            }
        }

        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<'a, 'b> PartialOrd<$lhs> for $rhs {
            #[inline]
            fn partial_cmp(&self, other: &$lhs) -> Option<cmp::Ordering> {
                <Path as PartialOrd>::partial_cmp(self, other)
            }
        }
    };
}

impl_cmp!(PathBuf, Path);
impl_cmp!(PathBuf, &'a Path);
impl_cmp!(Cow<'a, Path>, Path);
impl_cmp!(Cow<'a, Path>, &'b Path);
impl_cmp!(Cow<'a, Path>, PathBuf);

macro_rules! impl_cmp_os_str {
    ($lhs:ty, $rhs: ty) => {
        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<'a, 'b> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                <Path as PartialEq>::eq(self, other.as_ref())
            }
        }

        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<'a, 'b> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                <Path as PartialEq>::eq(self.as_ref(), other)
            }
        }

        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<'a, 'b> PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<cmp::Ordering> {
                <Path as PartialOrd>::partial_cmp(self, other.as_ref())
            }
        }

        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<'a, 'b> PartialOrd<$lhs> for $rhs {
            #[inline]
            fn partial_cmp(&self, other: &$lhs) -> Option<cmp::Ordering> {
                <Path as PartialOrd>::partial_cmp(self.as_ref(), other)
            }
        }
    };
}

impl_cmp_os_str!(PathBuf, OsStr);
impl_cmp_os_str!(PathBuf, &'a OsStr);
impl_cmp_os_str!(PathBuf, Cow<'a, OsStr>);
impl_cmp_os_str!(PathBuf, OsString);
impl_cmp_os_str!(Path, OsStr);
impl_cmp_os_str!(Path, &'a OsStr);
impl_cmp_os_str!(Path, Cow<'a, OsStr>);
impl_cmp_os_str!(Path, OsString);
impl_cmp_os_str!(&'a Path, OsStr);
impl_cmp_os_str!(&'a Path, Cow<'b, OsStr>);
impl_cmp_os_str!(&'a Path, OsString);
impl_cmp_os_str!(Cow<'a, Path>, OsStr);
impl_cmp_os_str!(Cow<'a, Path>, &'b OsStr);
impl_cmp_os_str!(Cow<'a, Path>, OsString);

#[stable(since = "1.7.0", feature = "strip_prefix")]
impl fmt::Display for StripPrefixError {
    #[allow(deprecated, deprecated_in_future)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.description().fmt(f)
    }
}

#[stable(since = "1.7.0", feature = "strip_prefix")]
impl Error for StripPrefixError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "prefix not found"
    }
}
