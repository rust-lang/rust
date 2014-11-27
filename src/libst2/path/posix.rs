// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! POSIX file path handling

use c_str::{CString, ToCStr};
use clone::Clone;
use cmp::{PartialEq, Eq, PartialOrd, Ord, Ordering};
use hash;
use io::Writer;
use iter::{DoubleEndedIterator, AdditiveIterator, Extend, Iterator, Map};
use kinds::Sized;
use option::{Option, None, Some};
use str::{FromStr, Str};
use str;
use slice::{CloneSliceAllocPrelude, Splits, AsSlice, VectorVector,
            PartialEqSlicePrelude, SlicePrelude};
use vec::Vec;

use super::{BytesContainer, GenericPath, GenericPathUnsafe};

/// Iterator that yields successive components of a Path as &[u8]
pub type Components<'a> = Splits<'a, u8>;

/// Iterator that yields successive components of a Path as Option<&str>
pub type StrComponents<'a> = Map<'a, &'a [u8], Option<&'a str>,
                                       Components<'a>>;

/// Represents a POSIX file path
#[deriving(Clone)]
pub struct Path {
    repr: Vec<u8>, // assumed to never be empty or contain NULs
    sepidx: Option<uint> // index of the final separator in repr
}

/// The standard path separator character
pub const SEP: char = '/';

/// The standard path separator byte
pub const SEP_BYTE: u8 = SEP as u8;

/// Returns whether the given byte is a path separator
#[inline]
pub fn is_sep_byte(u: &u8) -> bool { unimplemented!() }

/// Returns whether the given char is a path separator
#[inline]
pub fn is_sep(c: char) -> bool { unimplemented!() }

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
    #[inline]
    fn hash(&self, state: &mut S) { unimplemented!() }
}

impl BytesContainer for Path {
    #[inline]
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] { unimplemented!() }
}

impl GenericPathUnsafe for Path {
    unsafe fn new_unchecked<T: BytesContainer>(path: T) -> Path { unimplemented!() }

    unsafe fn set_filename_unchecked<T: BytesContainer>(&mut self, filename: T) { unimplemented!() }

    unsafe fn push_unchecked<T: BytesContainer>(&mut self, path: T) { unimplemented!() }
}

impl GenericPath for Path {
    #[inline]
    fn as_vec<'a>(&'a self) -> &'a [u8] { unimplemented!() }

    fn into_vec(self) -> Vec<u8> { unimplemented!() }

    fn dirname<'a>(&'a self) -> &'a [u8] { unimplemented!() }

    fn filename<'a>(&'a self) -> Option<&'a [u8]> { unimplemented!() }

    fn pop(&mut self) -> bool { unimplemented!() }

    fn root_path(&self) -> Option<Path> { unimplemented!() }

    #[inline]
    fn is_absolute(&self) -> bool { unimplemented!() }

    fn is_ancestor_of(&self, other: &Path) -> bool { unimplemented!() }

    fn path_relative_from(&self, base: &Path) -> Option<Path> { unimplemented!() }

    fn ends_with_path(&self, child: &Path) -> bool { unimplemented!() }
}

impl Path {
    /// Returns a new Path from a byte vector or string
    ///
    /// # Panics
    ///
    /// Panics the task if the vector contains a NUL.
    #[inline]
    pub fn new<T: BytesContainer>(path: T) -> Path { unimplemented!() }

    /// Returns a new Path from a byte vector or string, if possible
    #[inline]
    pub fn new_opt<T: BytesContainer>(path: T) -> Option<Path> { unimplemented!() }

    /// Returns a normalized byte vector representation of a path, by removing all empty
    /// components, and unnecessary . and .. components.
    fn normalize<Sized? V: AsSlice<u8>>(v: &V) -> Vec<u8> { unimplemented!() }

    /// Returns an iterator that yields each component of the path in turn.
    /// Does not distinguish between absolute and relative paths, e.g.
    /// /a/b/c and a/b/c yield the same set of components.
    /// A path of "/" yields no components. A path of "." yields one component.
    pub fn components<'a>(&'a self) -> Components<'a> { unimplemented!() }

    /// Returns an iterator that yields each component of the path as Option<&str>.
    /// See components() for details.
    pub fn str_components<'a>(&'a self) -> StrComponents<'a> { unimplemented!() }
}

// None result means the byte vector didn't need normalizing
fn normalize_helper<'a>(v: &'a [u8], is_abs: bool) -> Option<Vec<&'a [u8]>> { unimplemented!() }

#[allow(non_upper_case_globals)]
static dot_static: &'static [u8] = b".";
#[allow(non_upper_case_globals)]
static dot_dot_static: &'static [u8] = b"..";
