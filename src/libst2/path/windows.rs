// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15883

//! Windows file path handling

pub use self::PathPrefix::*;

use ascii::AsciiCast;
use c_str::{CString, ToCStr};
use clone::Clone;
use cmp::{PartialEq, Eq, PartialOrd, Ord, Ordering};
use hash;
use io::Writer;
use iter::{AdditiveIterator, DoubleEndedIterator, Extend, Iterator, Map};
use mem;
use option::{Option, Some, None};
use slice::{AsSlice, SlicePrelude};
use str::{CharSplits, FromStr, Str, StrAllocating, StrVector, StrPrelude};
use string::String;
use unicode::char::UnicodeChar;
use vec::Vec;

use super::{contains_nul, BytesContainer, GenericPath, GenericPathUnsafe};

/// Iterator that yields successive components of a Path as &str
///
/// Each component is yielded as Option<&str> for compatibility with PosixPath, but
/// every component in WindowsPath is guaranteed to be Some.
pub type StrComponents<'a> = Map<'a, &'a str, Option<&'a str>,
                                       CharSplits<'a, char>>;

/// Iterator that yields successive components of a Path as &[u8]
pub type Components<'a> = Map<'a, Option<&'a str>, &'a [u8],
                                    StrComponents<'a>>;

/// Represents a Windows path
// Notes for Windows path impl:
// The MAX_PATH is 260, but 253 is the practical limit due to some API bugs
// See http://msdn.microsoft.com/en-us/library/windows/desktop/aa365247.aspx for good information
// about windows paths.
// That same page puts a bunch of restrictions on allowed characters in a path.
// `\foo.txt` means "relative to current drive", but will not be considered to be absolute here
// as `∃P | P.join("\foo.txt") != "\foo.txt"`.
// `C:` is interesting, that means "the current directory on drive C".
// Long absolute paths need to have \\?\ prefix (or, for UNC, \\?\UNC\). I think that can be
// ignored for now, though, and only added in a hypothetical .to_pwstr() function.
// However, if a path is parsed that has \\?\, this needs to be preserved as it disables the
// processing of "." and ".." components and / as a separator.
// Experimentally, \\?\foo is not the same thing as \foo.
// Also, \\foo is not valid either (certainly not equivalent to \foo).
// Similarly, C:\\Users is not equivalent to C:\Users, although C:\Users\\foo is equivalent
// to C:\Users\foo. In fact the command prompt treats C:\\foo\bar as UNC path. But it might be
// best to just ignore that and normalize it to C:\foo\bar.
//
// Based on all this, I think the right approach is to do the following:
// * Require valid utf-8 paths. Windows API may use WCHARs, but we don't, and utf-8 is convertible
// to UTF-16 anyway (though does Windows use UTF-16 or UCS-2? Not sure).
// * Parse the prefixes \\?\UNC\, \\?\, and \\.\ explicitly.
// * If \\?\UNC\, treat following two path components as server\share. Don't error for missing
//   server\share.
// * If \\?\, parse disk from following component, if present. Don't error for missing disk.
// * If \\.\, treat rest of path as just regular components. I don't know how . and .. are handled
//   here, they probably aren't, but I'm not going to worry about that.
// * Else if starts with \\, treat following two components as server\share. Don't error for missing
//   server\share.
// * Otherwise, attempt to parse drive from start of path.
//
// The only error condition imposed here is valid utf-8. All other invalid paths are simply
// preserved by the data structure; let the Windows API error out on them.
#[deriving(Clone)]
pub struct Path {
    repr: String, // assumed to never be empty
    prefix: Option<PathPrefix>,
    sepidx: Option<uint> // index of the final separator in the non-prefix portion of repr
}

impl PartialEq for Path {
    #[inline]
    fn eq(&self, other: &Path) -> bool { unimplemented!() }
}

impl Eq for Path {}

impl PartialOrd for Path {
    fn partial_cmp(&self, other: &Path) -> Option<Ordering> { unimplemented!() }
}

impl Ord for Path {
    fn cmp(&self, other: &Path) -> Ordering { unimplemented!() }
}

impl FromStr for Path {
    fn from_str(s: &str) -> Option<Path> { unimplemented!() }
}

// FIXME (#12938): Until DST lands, we cannot decompose &str into & and str, so
// we cannot usefully take ToCStr arguments by reference (without forcing an
// additional & around &str). So we are instead temporarily adding an instance
// for &Path, so that we can take ToCStr as owned. When DST lands, the &Path
// instance should be removed, and arguments bound by ToCStr should be passed by
// reference.

impl ToCStr for Path {
    #[inline]
    fn to_c_str(&self) -> CString { unimplemented!() }

    #[inline]
    unsafe fn to_c_str_unchecked(&self) -> CString { unimplemented!() }
}

impl<S: hash::Writer> hash::Hash<S> for Path {
    #[cfg(not(test))]
    #[inline]
    fn hash(&self, state: &mut S) { unimplemented!() }

    #[cfg(test)]
    #[inline]
    fn hash(&self, _: &mut S) { unimplemented!() }
}

impl BytesContainer for Path {
    #[inline]
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] { unimplemented!() }
    #[inline]
    fn container_as_str<'a>(&'a self) -> Option<&'a str> { unimplemented!() }
    #[inline]
    fn is_str(_: Option<&Path>) -> bool { unimplemented!() }
}

impl GenericPathUnsafe for Path {
    /// See `GenericPathUnsafe::from_vec_unchecked`.
    ///
    /// # Panics
    ///
    /// Panics if not valid UTF-8.
    #[inline]
    unsafe fn new_unchecked<T: BytesContainer>(path: T) -> Path { unimplemented!() }

    /// See `GenericPathUnsafe::set_filename_unchecked`.
    ///
    /// # Panics
    ///
    /// Panics if not valid UTF-8.
    unsafe fn set_filename_unchecked<T: BytesContainer>(&mut self, filename: T) { unimplemented!() }

    /// See `GenericPathUnsafe::push_unchecked`.
    ///
    /// Concatenating two Windows Paths is rather complicated.
    /// For the most part, it will behave as expected, except in the case of
    /// pushing a volume-relative path, e.g. `C:foo.txt`. Because we have no
    /// concept of per-volume cwds like Windows does, we can't behave exactly
    /// like Windows will. Instead, if the receiver is an absolute path on
    /// the same volume as the new path, it will be treated as the cwd that
    /// the new path is relative to. Otherwise, the new path will be treated
    /// as if it were absolute and will replace the receiver outright.
    unsafe fn push_unchecked<T: BytesContainer>(&mut self, path: T) { unimplemented!() }
}

impl GenericPath for Path {
    #[inline]
    fn new_opt<T: BytesContainer>(path: T) -> Option<Path> { unimplemented!() }

    /// See `GenericPath::as_str` for info.
    /// Always returns a `Some` value.
    #[inline]
    fn as_str<'a>(&'a self) -> Option<&'a str> { unimplemented!() }

    #[inline]
    fn as_vec<'a>(&'a self) -> &'a [u8] { unimplemented!() }

    #[inline]
    fn into_vec(self) -> Vec<u8> { unimplemented!() }

    #[inline]
    fn dirname<'a>(&'a self) -> &'a [u8] { unimplemented!() }

    /// See `GenericPath::dirname_str` for info.
    /// Always returns a `Some` value.
    fn dirname_str<'a>(&'a self) -> Option<&'a str> { unimplemented!() }

    #[inline]
    fn filename<'a>(&'a self) -> Option<&'a [u8]> { unimplemented!() }

    /// See `GenericPath::filename_str` for info.
    /// Always returns a `Some` value if `filename` returns a `Some` value.
    fn filename_str<'a>(&'a self) -> Option<&'a str> { unimplemented!() }

    /// See `GenericPath::filestem_str` for info.
    /// Always returns a `Some` value if `filestem` returns a `Some` value.
    #[inline]
    fn filestem_str<'a>(&'a self) -> Option<&'a str> { unimplemented!() }

    #[inline]
    fn extension_str<'a>(&'a self) -> Option<&'a str> { unimplemented!() }

    fn dir_path(&self) -> Path { unimplemented!() }

    #[inline]
    fn pop(&mut self) -> bool { unimplemented!() }

    fn root_path(&self) -> Option<Path> { unimplemented!() }

    /// See `GenericPath::is_absolute` for info.
    ///
    /// A Windows Path is considered absolute only if it has a non-volume prefix,
    /// or if it has a volume prefix and the path starts with '\'.
    /// A path of `\foo` is not considered absolute because it's actually
    /// relative to the "current volume". A separate method `Path::is_vol_relative`
    /// is provided to indicate this case. Similarly a path of `C:foo` is not
    /// considered absolute because it's relative to the cwd on volume C:. A
    /// separate method `Path::is_cwd_relative` is provided to indicate this case.
    #[inline]
    fn is_absolute(&self) -> bool { unimplemented!() }

    #[inline]
    fn is_relative(&self) -> bool { unimplemented!() }

    fn is_ancestor_of(&self, other: &Path) -> bool { unimplemented!() }

    fn path_relative_from(&self, base: &Path) -> Option<Path> { unimplemented!() }

    fn ends_with_path(&self, child: &Path) -> bool { unimplemented!() }
}

impl Path {
    /// Returns a new `Path` from a `BytesContainer`.
    ///
    /// # Panics
    ///
    /// Panics if the vector contains a `NUL`, or if it contains invalid UTF-8.
    ///
    /// # Example
    ///
    /// ```
    /// println!("{}", Path::new(r"C:\some\path").display());
    /// ```
    #[inline]
    pub fn new<T: BytesContainer>(path: T) -> Path { unimplemented!() }

    /// Returns a new `Some(Path)` from a `BytesContainer`.
    ///
    /// Returns `None` if the vector contains a `NUL`, or if it contains invalid UTF-8.
    ///
    /// # Example
    ///
    /// ```
    /// let path = Path::new_opt(r"C:\some\path");
    ///
    /// match path {
    ///     Some(path) => println!("{}", path.display()),
    ///     None       => println!("There was a problem with your path."),
    /// }
    /// ```
    #[inline]
    pub fn new_opt<T: BytesContainer>(path: T) -> Option<Path> { unimplemented!() }

    /// Returns an iterator that yields each component of the path in turn as a Option<&str>.
    /// Every component is guaranteed to be Some.
    /// Does not yield the path prefix (including server/share components in UNC paths).
    /// Does not distinguish between volume-relative and relative paths, e.g.
    /// \a\b\c and a\b\c.
    /// Does not distinguish between absolute and cwd-relative paths, e.g.
    /// C:\foo and C:foo.
    pub fn str_components<'a>(&'a self) -> StrComponents<'a> { unimplemented!() }

    /// Returns an iterator that yields each component of the path in turn as a &[u8].
    /// See str_components() for details.
    pub fn components<'a>(&'a self) -> Components<'a> { unimplemented!() }

    fn equiv_prefix(&self, other: &Path) -> bool { unimplemented!() }

    fn normalize_<S: StrAllocating>(s: S) -> (Option<PathPrefix>, String) { unimplemented!() }

    fn normalize__(s: &str, prefix: Option<PathPrefix>) -> Option<String> { unimplemented!() }

    fn update_sepidx(&mut self) { unimplemented!() }

    fn prefix_len(&self) -> uint { unimplemented!() }

    // Returns a tuple (before, after, end) where before is the index of the separator
    // and after is the index just after the separator.
    // end is the length of the string, normally, or the index of the final character if it is
    // a non-semantic trailing separator in a verbatim string.
    // If the prefix is considered the separator, before and after are the same.
    fn sepidx_or_prefix_len(&self) -> Option<(uint,uint,uint)> { unimplemented!() }

    fn has_nonsemantic_trailing_slash(&self) -> bool { unimplemented!() }

    fn update_normalized<S: Str>(&mut self, s: S) { unimplemented!() }
}

/// Returns whether the path is considered "volume-relative", which means a path
/// that looks like "\foo". Paths of this form are relative to the current volume,
/// but absolute within that volume.
#[inline]
pub fn is_vol_relative(path: &Path) -> bool { unimplemented!() }

/// Returns whether the path is considered "cwd-relative", which means a path
/// with a volume prefix that is not absolute. This look like "C:foo.txt". Paths
/// of this form are relative to the cwd on the given volume.
#[inline]
pub fn is_cwd_relative(path: &Path) -> bool { unimplemented!() }

/// Returns the PathPrefix for this Path
#[inline]
pub fn prefix(path: &Path) -> Option<PathPrefix> { unimplemented!() }

/// Returns whether the Path's prefix is a verbatim prefix, i.e. `\\?\`
#[inline]
pub fn is_verbatim(path: &Path) -> bool { unimplemented!() }

/// Returns the non-verbatim equivalent of the input path, if possible.
/// If the input path is a device namespace path, None is returned.
/// If the input path is not verbatim, it is returned as-is.
/// If the input path is verbatim, but the same path can be expressed as
/// non-verbatim, the non-verbatim version is returned.
/// Otherwise, None is returned.
pub fn make_non_verbatim(path: &Path) -> Option<Path> { unimplemented!() }

/// The standard path separator character
pub const SEP: char = '\\';
/// The standard path separator byte
pub const SEP_BYTE: u8 = SEP as u8;

/// The alternative path separator character
pub const SEP2: char = '/';
/// The alternative path separator character
pub const SEP2_BYTE: u8 = SEP2 as u8;

/// Returns whether the given char is a path separator.
/// Allows both the primary separator '\' and the alternative separator '/'.
#[inline]
pub fn is_sep(c: char) -> bool { unimplemented!() }

/// Returns whether the given char is a path separator.
/// Only allows the primary separator '\'; use is_sep to allow '/'.
#[inline]
pub fn is_sep_verbatim(c: char) -> bool { unimplemented!() }

/// Returns whether the given byte is a path separator.
/// Allows both the primary separator '\' and the alternative separator '/'.
#[inline]
pub fn is_sep_byte(u: &u8) -> bool { unimplemented!() }

/// Returns whether the given byte is a path separator.
/// Only allows the primary separator '\'; use is_sep_byte to allow '/'.
#[inline]
pub fn is_sep_byte_verbatim(u: &u8) -> bool { unimplemented!() }

/// Prefix types for Path
#[deriving(PartialEq, Clone, Show)]
pub enum PathPrefix {
    /// Prefix `\\?\`, uint is the length of the following component
    VerbatimPrefix(uint),
    /// Prefix `\\?\UNC\`, uints are the lengths of the UNC components
    VerbatimUNCPrefix(uint, uint),
    /// Prefix `\\?\C:\` (for any alphabetic character)
    VerbatimDiskPrefix,
    /// Prefix `\\.\`, uint is the length of the following component
    DeviceNSPrefix(uint),
    /// UNC prefix `\\server\share`, uints are the lengths of the server/share
    UNCPrefix(uint, uint),
    /// Prefix `C:` for any alphabetic character
    DiskPrefix
}

fn parse_prefix<'a>(mut path: &'a str) -> Option<PathPrefix> { unimplemented!() }

// None result means the string didn't need normalizing
fn normalize_helper<'a>(s: &'a str, prefix: Option<PathPrefix>) -> (bool, Option<Vec<&'a str>>) { unimplemented!() }

fn prefix_is_verbatim(p: Option<PathPrefix>) -> bool { unimplemented!() }

fn prefix_len(p: Option<PathPrefix>) -> uint { unimplemented!() }
