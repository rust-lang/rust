// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A structure for holding a set of enum variants.
//!
//! This module defines a container which uses an efficient bit mask
//! representation to hold C-like enum variants.

#![unstable(feature = "enumset",
            reason = "matches collection reform specification, \
                      waiting for dust to settle",
            issue = "37966")]

use core::marker;
use core::fmt;
use core::iter::{FromIterator, FusedIterator};
use core::ops::{Sub, BitOr, BitAnd, BitXor};

// FIXME(contentions): implement union family of methods? (general design may be
// wrong here)

/// A specialized set implementation to use enum types.
///
/// It is a logic error for an item to be modified in such a way that the
/// transformation of the item to or from a `usize`, as determined by the
/// `CLike` trait, changes while the item is in the set. This is normally only
/// possible through `Cell`, `RefCell`, global state, I/O, or unsafe code.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnumSet<E> {
    // We must maintain the invariant that no bits are set
    // for which no variant exists
    bits: usize,
    marker: marker::PhantomData<E>,
}

impl<E> Copy for EnumSet<E> {}

impl<E> Clone for EnumSet<E> {
    fn clone(&self) -> EnumSet<E> {
        *self
    }
}

impl<E: CLike + fmt::Debug> fmt::Debug for EnumSet<E> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_set().entries(self).finish()
    }
}

/// An interface for casting C-like enum to usize and back.
/// A typically implementation is as below.
///
/// ```{rust,ignore}
/// #[repr(usize)]
/// enum Foo {
///     A, B, C
/// }
///
/// impl CLike for Foo {
///     fn to_usize(&self) -> usize {
///         *self as usize
///     }
///
///     fn from_usize(v: usize) -> Foo {
///         unsafe { mem::transmute(v) }
///     }
/// }
/// ```
pub trait CLike {
    /// Converts a C-like enum to a `usize`.
    fn to_usize(&self) -> usize;
    /// Converts a `usize` to a C-like enum.
    fn from_usize(usize) -> Self;
}

fn bit<E: CLike>(e: &E) -> usize {
    use core::mem;
    let value = e.to_usize();
    let bits = mem::size_of::<usize>() * 8;
    assert!(value < bits,
            "EnumSet only supports up to {} variants.",
            bits - 1);
    1 << value
}

impl<E: CLike> EnumSet<E> {
    /// Returns an empty `EnumSet`.
    pub fn new() -> EnumSet<E> {
        EnumSet {
            bits: 0,
            marker: marker::PhantomData,
        }
    }

    /// Returns the number of elements in the given `EnumSet`.
    pub fn len(&self) -> usize {
        self.bits.count_ones() as usize
    }

    /// Returns true if the `EnumSet` is empty.
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    pub fn clear(&mut self) {
        self.bits = 0;
    }

    /// Returns `false` if the `EnumSet` contains any enum of the given `EnumSet`.
    pub fn is_disjoint(&self, other: &EnumSet<E>) -> bool {
        (self.bits & other.bits) == 0
    }

    /// Returns `true` if a given `EnumSet` is included in this `EnumSet`.
    pub fn is_superset(&self, other: &EnumSet<E>) -> bool {
        (self.bits & other.bits) == other.bits
    }

    /// Returns `true` if this `EnumSet` is included in the given `EnumSet`.
    pub fn is_subset(&self, other: &EnumSet<E>) -> bool {
        other.is_superset(self)
    }

    /// Returns the union of both `EnumSets`.
    pub fn union(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {
            bits: self.bits | e.bits,
            marker: marker::PhantomData,
        }
    }

    /// Returns the intersection of both `EnumSets`.
    pub fn intersection(&self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {
            bits: self.bits & e.bits,
            marker: marker::PhantomData,
        }
    }

    /// Adds an enum to the `EnumSet`, and returns `true` if it wasn't there before
    pub fn insert(&mut self, e: E) -> bool {
        let result = !self.contains(&e);
        self.bits |= bit(&e);
        result
    }

    /// Removes an enum from the EnumSet
    pub fn remove(&mut self, e: &E) -> bool {
        let result = self.contains(e);
        self.bits &= !bit(e);
        result
    }

    /// Returns `true` if an `EnumSet` contains a given enum.
    pub fn contains(&self, e: &E) -> bool {
        (self.bits & bit(e)) != 0
    }

    /// Returns an iterator over an `EnumSet`.
    pub fn iter(&self) -> Iter<E> {
        Iter::new(self.bits)
    }
}

impl<E: CLike> Sub for EnumSet<E> {
    type Output = EnumSet<E>;

    fn sub(self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {
            bits: self.bits & !e.bits,
            marker: marker::PhantomData,
        }
    }
}

impl<E: CLike> BitOr for EnumSet<E> {
    type Output = EnumSet<E>;

    fn bitor(self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {
            bits: self.bits | e.bits,
            marker: marker::PhantomData,
        }
    }
}

impl<E: CLike> BitAnd for EnumSet<E> {
    type Output = EnumSet<E>;

    fn bitand(self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {
            bits: self.bits & e.bits,
            marker: marker::PhantomData,
        }
    }
}

impl<E: CLike> BitXor for EnumSet<E> {
    type Output = EnumSet<E>;

    fn bitxor(self, e: EnumSet<E>) -> EnumSet<E> {
        EnumSet {
            bits: self.bits ^ e.bits,
            marker: marker::PhantomData,
        }
    }
}

/// An iterator over an EnumSet
pub struct Iter<E> {
    index: usize,
    bits: usize,
    marker: marker::PhantomData<E>,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<E> Clone for Iter<E> {
    fn clone(&self) -> Iter<E> {
        Iter {
            index: self.index,
            bits: self.bits,
            marker: marker::PhantomData,
        }
    }
}

impl<E: CLike> Iter<E> {
    fn new(bits: usize) -> Iter<E> {
        Iter {
            index: 0,
            bits: bits,
            marker: marker::PhantomData,
        }
    }
}

impl<E: CLike> Iterator for Iter<E> {
    type Item = E;

    fn next(&mut self) -> Option<E> {
        if self.bits == 0 {
            return None;
        }

        while (self.bits & 1) == 0 {
            self.index += 1;
            self.bits >>= 1;
        }
        let elem = CLike::from_usize(self.index);
        self.index += 1;
        self.bits >>= 1;
        Some(elem)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.bits.count_ones() as usize;
        (exact, Some(exact))
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<E: CLike> FusedIterator for Iter<E> {}

impl<E: CLike> FromIterator<E> for EnumSet<E> {
    fn from_iter<I: IntoIterator<Item = E>>(iter: I) -> EnumSet<E> {
        let mut ret = EnumSet::new();
        ret.extend(iter);
        ret
    }
}

impl<'a, E> IntoIterator for &'a EnumSet<E>
    where E: CLike
{
    type Item = E;
    type IntoIter = Iter<E>;

    fn into_iter(self) -> Iter<E> {
        self.iter()
    }
}

impl<E: CLike> Extend<E> for EnumSet<E> {
    fn extend<I: IntoIterator<Item = E>>(&mut self, iter: I) {
        for element in iter {
            self.insert(element);
        }
    }
}

impl<'a, E: 'a + CLike + Copy> Extend<&'a E> for EnumSet<E> {
    fn extend<I: IntoIterator<Item = &'a E>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}
