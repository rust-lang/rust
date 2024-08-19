#![deny(unsafe_op_in_unsafe_fn)]

use crate::clone::CloneToUninit;
use crate::error::Error;
use crate::ffi::os_str::{self, OsStr};
use crate::hash::{Hash, Hasher};
use crate::iter::FusedIterator;
use crate::{cmp, fmt};

#[cfg(target_os = "windows")]
mod windows;
#[unstable(
    feature = "path_internals",
    reason = "internal details of the implementation of path",
    issue = "none"
)]
#[doc(hidden)]
#[cfg(target_os = "windows")]
pub use windows::*;

#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
mod sgx;
#[unstable(
    feature = "path_internals",
    reason = "internal details of the implementation of path",
    issue = "none"
)]
#[doc(hidden)]
#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
pub use sgx::*;

#[cfg(any(target_os = "uefi", target_os = "solid_asp3",))]
mod unsupported_backslash;
#[unstable(
    feature = "path_internals",
    reason = "internal details of the implementation of path",
    issue = "none"
)]
#[doc(hidden)]
#[cfg(any(target_os = "uefi", target_os = "solid_asp3",))]
pub use unsupported_backslash::*;

#[cfg(unix)]
mod unix;
#[unstable(
    feature = "path_internals",
    reason = "internal details of the implementation of path",
    issue = "none"
)]
#[doc(hidden)]
#[cfg(unix)]
pub use unix::*;

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
/// fn get_path_prefix(s: &str) -> Prefix<'_> {
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
    /// Device namespace prefixes consist of `\\.\` (possibly using `/`
    /// instead of `\`), immediately followed by the device name.
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
    /// Length
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn len(&self) -> usize {
        use self::Prefix::*;
        fn os_str_len(s: &OsStr) -> usize {
            s.as_encoded_bytes().len()
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_verbatim(&self) -> bool {
        use self::Prefix::*;
        matches!(*self, Verbatim(_) | VerbatimDisk(_) | VerbatimUNC(..))
    }

    /// Is drive
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn is_drive(&self) -> bool {
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
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn is_separator(c: char) -> bool {
    c.is_ascii() && is_sep_byte(c as u8)
}

/// The primary separator of path components for the current platform.
///
/// For example, `/` on Unix and `\` on Windows.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAIN_SEPARATOR: char = MAIN_SEP;

/// The primary separator of path components for the current platform.
///
/// For example, `/` on Unix and `\` on Windows.
#[stable(feature = "main_separator_str", since = "1.68.0")]
pub const MAIN_SEPARATOR_STR: &str = MAIN_SEP_STR;

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
    if file.as_encoded_bytes() == b".." {
        return (Some(file), None);
    }

    let mut iter = file.as_encoded_bytes().rsplitn(2, |b| *b == b'.');
    let after = iter.next();
    let before = iter.next();
    if before == Some(b"") {
        (Some(file), None)
    } else {
        // SAFETY:
        // The unsafety here stems from converting between &OsStr and &[u8]
        // and back. This is safe to do because (1) we only look at ASCII
        // contents of the encoding and (2) new &OsStr values are produced
        // only from ASCII-bounded slices of existing &OsStr values.
        unsafe {
            (
                before.map(|s| OsStr::from_encoded_bytes_unchecked(s)),
                after.map(|s| OsStr::from_encoded_bytes_unchecked(s)),
            )
        }
    }
}

fn split_file_at_dot(file: &OsStr) -> (&OsStr, Option<&OsStr>) {
    let slice = file.as_encoded_bytes();
    if slice == b".." {
        return (file, None);
    }

    let i = match slice[1..].iter().position(|b| *b == b'.') {
        Some(i) => i + 1,
        None => return (file, None),
    };
    let before = &slice[..i];
    let after = &slice[i + 1..];
    // SAFETY:
    // The unsafety here stems from converting between &OsStr and &[u8]
    // and back. This is safe to do because (1) we only look at ASCII
    // contents of the encoding and (2) new &OsStr values are produced
    // only from ASCII-bounded slices of existing &OsStr values.
    unsafe {
        (
            OsStr::from_encoded_bytes_unchecked(before),
            Some(OsStr::from_encoded_bytes_unchecked(after)),
        )
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
    #[must_use]
    #[inline]
    pub fn kind(&self) -> Prefix<'a> {
        self.parsed
    }

    /// Returns the raw [`OsStr`] slice for this prefix.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub fn as_os_str(&self) -> &'a OsStr {
        self.raw
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> PartialEq for PrefixComponent<'a> {
    #[inline]
    fn eq(&self, other: &PrefixComponent<'a>) -> bool {
        self.parsed == other.parsed
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> PartialOrd for PrefixComponent<'a> {
    #[inline]
    fn partial_cmp(&self, other: &PrefixComponent<'a>) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&self.parsed, &other.parsed)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for PrefixComponent<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&self.parsed, &other.parsed)
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
///     println!("{component:?}");
/// }
/// ```
///
/// [`components`]: Path::components
#[derive(Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Components<'a> {
    // The path left to parse components from
    path: &'a [u8],

    // The prefix as it was originally parsed, if any
    prefix: Option<Prefix<'a>>,

    // true if path *physically* has a root separator; for most Windows
    // prefixes, it may have a "logical" root separator for the purposes of
    // normalization, e.g., \\server\share == \\server\share\.
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
#[must_use = "iterators are lazy and do nothing unless consumed"]
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
    // The path left to parse components from
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn path_left_to_parse(&self) -> &'a [u8] {
        self.path
    }

    // The prefix as it was originally parsed, if any
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn prefix(&self) -> Option<Prefix<'a>> {
        self.prefix
    }

    // how long is the prefix, if any?
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn prefix_len(&self) -> usize {
        self.prefix.as_ref().map(Prefix::len).unwrap_or(0)
    }

    /// Internal method
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn prefix_verbatim(&self) -> bool {
        self.prefix.as_ref().map(Prefix::is_verbatim).unwrap_or(false)
    }

    /// how much of the prefix is left from the point of view of iteration?
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn prefix_remaining(&self) -> usize {
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_path(&self) -> &'a Path {
        let mut comps = self.clone();
        if comps.front == State::Body {
            comps.trim_left();
        }
        if comps.back == State::Body {
            comps.trim_right();
        }
        // SAFETY: comps.path is valid `OsStr`
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
        let mut iter = self.path[self.prefix_remaining()..].iter();
        match (iter.next(), iter.next()) {
            (Some(&b'.'), None) => true,
            (Some(&b'.'), Some(&b)) => self.is_sep_byte(b),
            _ => false,
        }
    }

    // parse a given byte sequence following the OsStr encoding into the
    // corresponding path component
    unsafe fn parse_single_component<'b>(&self, comp: &'b [u8]) -> Option<Component<'b>> {
        match comp {
            b"." if self.prefix_verbatim() => Some(Component::CurDir),
            b"." => None, // . components are normalized away, except at
            // the beginning of a path, which is treated
            // separately via `include_cur_dir`
            b".." => Some(Component::ParentDir),
            b"" => None,
            // SAFETY: comp is valid `OsStr`
            _ => Some(Component::Normal(unsafe { OsStr::from_encoded_bytes_unchecked(comp) })),
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
        // SAFETY: `comp` is a valid substring, since it is split on a separator.
        (comp.len() + extra, unsafe { self.parse_single_component(comp) })
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
        // SAFETY: `comp` is a valid substring, since it is split on a separator.
        (comp.len() + extra, unsafe { self.parse_single_component(comp) })
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
    #[must_use]
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
                        // SAFETY: raw is valid `OsStr`
                        raw: unsafe { OsStr::from_encoded_bytes_unchecked(raw) },
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
                        // SAFETY: self.path is valid `OsStr`
                        raw: unsafe { OsStr::from_encoded_bytes_unchecked(self.path) },
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
impl<'a> PartialEq for Components<'a> {
    #[inline]
    fn eq(&self, other: &Components<'a>) -> bool {
        let Components { path: _, front: _, back: _, has_physical_root: _, prefix: _ } = self;

        // Fast path for exact matches, e.g. for hashmap lookups.
        // Don't explicitly compare the prefix or has_physical_root fields since they'll
        // either be covered by the `path` buffer or are only relevant for `prefix_verbatim()`.
        if self.path.len() == other.path.len()
            && self.front == other.front
            && self.back == State::Body
            && other.back == State::Body
            && self.prefix_verbatim() == other.prefix_verbatim()
        {
            // possible future improvement: this could bail out earlier if there were a
            // reverse memcmp/bcmp comparing back to front
            if self.path == other.path {
                return true;
            }
        }

        // compare back to front since absolute paths often share long prefixes
        Iterator::eq(self.clone().rev(), other.clone().rev())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for Components<'_> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> PartialOrd for Components<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Components<'a>) -> Option<cmp::Ordering> {
        Some(compare_components(self.clone(), other.clone()))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for Components<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        compare_components(self.clone(), other.clone())
    }
}

#[unstable(
    feature = "path_internals",
    reason = "internal details of the implementation of path",
    issue = "none"
)]
#[doc(hidden)]
pub fn compare_components(mut left: Components<'_>, mut right: Components<'_>) -> cmp::Ordering {
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
        // possible future improvement: a [u8]::first_mismatch simd implementation
        let first_difference = match left.path.iter().zip(right.path).position(|(&a, &b)| a != b) {
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
#[must_use = "iterators are lazy and do nothing unless consumed"]
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
#[rustc_has_incoherent_inherent_impls]
// `Path::new` current implementation relies
// on `Path` being layout-compatible with `OsStr`.
// However, `Path` layout is considered an implementation detail and must not be relied upon.
#[repr(transparent)]
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
        // SAFETY: s must be a valid `OsStr`
        unsafe { Path::new(OsStr::from_encoded_bytes_unchecked(s)) }
    }

    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    /// The following (private!) function reveals the byte encoding used for OsStr.
    pub fn as_u8_slice(&self) -> &[u8] {
        self.inner.as_encoded_bytes()
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
        // SAFETY: `Path` is just a wrapper of `OsStr`
        unsafe { &*(s.as_ref() as *const OsStr as *const Path) }
    }

    /// From mutable `OsStr`
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    pub fn from_inner_mut(inner: &mut OsStr) -> &mut Path {
        // SAFETY: Path is just a wrapper around OsStr,
        // therefore converting &mut OsStr to &mut Path is safe.
        unsafe { &mut *(inner as *mut OsStr as *mut Path) }
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
    #[must_use]
    #[inline]
    pub fn as_os_str(&self) -> &OsStr {
        &self.inner
    }

    /// Yields a mutable reference to the underlying [`OsStr`] slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::{Path, PathBuf};
    ///
    /// let mut path = PathBuf::from("Foo.TXT");
    ///
    /// assert_ne!(path, Path::new("foo.txt"));
    ///
    /// path.as_mut_os_str().make_ascii_lowercase();
    /// assert_eq!(path, Path::new("foo.txt"));
    /// ```
    #[stable(feature = "path_as_mut_os_str", since = "1.70.0")]
    #[must_use]
    #[inline]
    pub fn as_mut_os_str(&mut self) -> &mut OsStr {
        &mut self.inner
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
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn to_str(&self) -> Option<&str> {
        self.inner.to_str()
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
    #[must_use]
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
    #[must_use]
    #[inline]
    pub fn is_relative(&self) -> bool {
        !self.is_absolute()
    }

    /// Prefix
    #[unstable(
        feature = "path_internals",
        reason = "internal details of the implementation of path",
        issue = "none"
    )]
    #[doc(hidden)]
    pub fn prefix(&self) -> Option<Prefix<'_>> {
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
    #[must_use]
    #[inline]
    pub fn has_root(&self) -> bool {
        self.components().has_root()
    }

    /// Returns the `Path` without its final component, if there is one.
    ///
    /// This means it returns `Some("")` for relative paths with one component.
    ///
    /// Returns [`None`] if the path terminates in a root or prefix, or if it's
    /// the empty string.
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
    ///
    /// let relative_path = Path::new("foo/bar");
    /// let parent = relative_path.parent();
    /// assert_eq!(parent, Some(Path::new("foo")));
    /// let grand_parent = parent.and_then(Path::parent);
    /// assert_eq!(grand_parent, Some(Path::new("")));
    /// let great_grand_parent = grand_parent.and_then(Path::parent);
    /// assert_eq!(great_grand_parent, None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[doc(alias = "dirname")]
    #[must_use]
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
    /// or more times. If the [`parent`] method returns [`None`], the iterator will do likewise.
    /// The iterator will always yield at least one value, namely `Some(&self)`. Next it will yield
    /// `&self.parent()`, `&self.parent().and_then(Path::parent)` and so on.
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
    #[doc(alias = "basename")]
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn file_prefix(&self) -> Option<&OsStr> {
        self.file_name().map(split_file_at_dot).and_then(|(before, _after)| Some(before))
    }

    /// Extracts the extension (without the leading dot) of [`self.file_name`], if possible.
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
    #[must_use]
    pub fn extension(&self) -> Option<&OsStr> {
        self.file_name().map(rsplit_file_at_dot).and_then(|(before, after)| before.and(after))
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
    /// [`Debug`]: fmt::Debug
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
        Display { inner: self.inner.display() }
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for Path {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut Self) {
        // SAFETY: Path is just a wrapper around OsStr
        unsafe { self.inner.clone_to_uninit(core::ptr::addr_of_mut!((*dst).inner)) }
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
    inner: os_str::Display<'a>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for Path {
    #[inline]
    fn eq(&self, other: &Path) -> bool {
        self.components() == other.components()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for Path {
    fn hash<H: Hasher>(&self, h: &mut H) {
        let bytes = self.as_u8_slice();
        let (prefix_len, verbatim) = match parse_prefix(&self.inner) {
            Some(prefix) => {
                prefix.hash(h);
                (prefix.len(), prefix.is_verbatim())
            }
            None => (0, false),
        };
        let bytes = &bytes[prefix_len..];

        let mut component_start = 0;
        // track some extra state to avoid prefix collisions.
        // ["foo", "bar"] and ["foobar"], will have the same payload bytes
        // but result in different chunk_bits
        let mut chunk_bits: usize = 0;

        for i in 0..bytes.len() {
            let is_sep = if verbatim { is_verbatim_sep(bytes[i]) } else { is_sep_byte(bytes[i]) };
            if is_sep {
                if i > component_start {
                    let to_hash = &bytes[component_start..i];
                    chunk_bits = chunk_bits.wrapping_add(to_hash.len());
                    chunk_bits = chunk_bits.rotate_right(2);
                    h.write(to_hash);
                }

                // skip over separator and optionally a following CurDir item
                // since components() would normalize these away.
                component_start = i + 1;

                let tail = &bytes[component_start..];

                if !verbatim {
                    component_start += match tail {
                        [b'.'] => 1,
                        [b'.', sep @ _, ..] if is_sep_byte(*sep) => 1,
                        _ => 0,
                    };
                }
            }
        }

        if component_start < bytes.len() {
            let to_hash = &bytes[component_start..];
            chunk_bits = chunk_bits.wrapping_add(to_hash.len());
            chunk_bits = chunk_bits.rotate_right(2);
            h.write(to_hash);
        }

        h.write_usize(chunk_bits);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for Path {}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for Path {
    #[inline]
    fn partial_cmp(&self, other: &Path) -> Option<cmp::Ordering> {
        Some(compare_components(self.components(), other.components()))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for Path {
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

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<Path> for str {
    #[inline]
    fn as_ref(&self) -> &Path {
        Path::new(self)
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

macro_rules! impl_cmp_os_str {
    (<$($life:lifetime),*> $lhs:ty, $rhs: ty) => {
        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<$($life),*> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                <Path as PartialEq>::eq(self, other.as_ref())
            }
        }

        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<$($life),*> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                <Path as PartialEq>::eq(self.as_ref(), other)
            }
        }

        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<$($life),*> PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<cmp::Ordering> {
                <Path as PartialOrd>::partial_cmp(self, other.as_ref())
            }
        }

        #[stable(feature = "cmp_path", since = "1.8.0")]
        impl<$($life),*> PartialOrd<$lhs> for $rhs {
            #[inline]
            fn partial_cmp(&self, other: &$lhs) -> Option<cmp::Ordering> {
                <Path as PartialOrd>::partial_cmp(self.as_ref(), other)
            }
        }
    };
}

impl_cmp_os_str!(<> Path, OsStr);
impl_cmp_os_str!(<'a> Path, &'a OsStr);
impl_cmp_os_str!(<'a> &'a Path, OsStr);

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
