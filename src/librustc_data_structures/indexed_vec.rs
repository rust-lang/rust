// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;
use std::iter::{self, FromIterator};
use std::slice;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range, RangeBounds};
use std::fmt;
use std::hash::Hash;
use std::vec;
use std::u32;

use rustc_serialize as serialize;

/// Represents some newtyped `usize` wrapper.
///
/// (purpose: avoid mixing indexes for different bitvector domains.)
pub trait Idx: Copy + 'static + Ord + Debug + Hash {
    fn new(idx: usize) -> Self;

    fn index(self) -> usize;

    fn increment_by(&mut self, amount: usize) {
        let v = self.index() + amount;
        *self = Self::new(v);
    }
}

impl Idx for usize {
    #[inline]
    fn new(idx: usize) -> Self { idx }
    #[inline]
    fn index(self) -> usize { self }
}

impl Idx for u32 {
    #[inline]
    fn new(idx: usize) -> Self { assert!(idx <= u32::MAX as usize); idx as u32 }
    #[inline]
    fn index(self) -> usize { self as usize }
}

#[macro_export]
macro_rules! newtype_index {
    // ---- public rules ----

    // Use default constants
    ($name:ident) => (
        newtype_index!(
            // Leave out derives marker so we can use its absence to ensure it comes first
            @type         [$name]
            @max          [::std::u32::MAX]
            @debug_format ["{}"]);
    );

    // Define any constants
    ($name:ident { $($tokens:tt)+ }) => (
        newtype_index!(
            // Leave out derives marker so we can use its absence to ensure it comes first
            @type         [$name]
            @max          [::std::u32::MAX]
            @debug_format ["{}"]
                          $($tokens)+);
    );

    // ---- private rules ----

    // Base case, user-defined constants (if any) have already been defined
    (@derives      [$($derives:ident,)*]
     @pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]) => (
        #[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, $($derives),*)]
        pub struct $type($($pub)* u32);

        impl Idx for $type {
            #[inline]
            fn new(value: usize) -> Self {
                assert!(value < ($max) as usize);
                $type(value as u32)
            }

            #[inline]
            fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl ::std::iter::Step for $type {
            fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                <usize as ::std::iter::Step>::steps_between(
                    &Idx::index(*start),
                    &Idx::index(*end),
                )
            }

            fn replace_one(&mut self) -> Self {
                ::std::mem::replace(self, Self::new(1))
            }

            fn replace_zero(&mut self) -> Self {
                ::std::mem::replace(self, Self::new(0))
            }

            fn add_one(&self) -> Self {
                Self::new(Idx::index(*self) + 1)
            }

            fn sub_one(&self) -> Self {
                Self::new(Idx::index(*self) - 1)
            }

            fn add_usize(&self, u: usize) -> Option<Self> {
                Idx::index(*self).checked_add(u).map(Self::new)
            }
        }

        newtype_index!(
            @handle_debug
            @derives      [$($derives,)*]
            @type         [$type]
            @debug_format [$debug_format]);
    );

    // base case for handle_debug where format is custom. No Debug implementation is emitted.
    (@handle_debug
     @derives      [$($_derives:ident,)*]
     @type         [$type:ident]
     @debug_format [custom]) => ();

    // base case for handle_debug, no debug overrides found, so use default
    (@handle_debug
     @derives      []
     @type         [$type:ident]
     @debug_format [$debug_format:tt]) => (
        impl ::std::fmt::Debug for $type {
            fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                write!(fmt, $debug_format, self.0)
            }
        }
    );

    // Debug is requested for derive, don't generate any Debug implementation.
    (@handle_debug
     @derives      [Debug, $($derives:ident,)*]
     @type         [$type:ident]
     @debug_format [$debug_format:tt]) => ();

    // It's not Debug, so just pop it off the front of the derives stack and check the rest.
    (@handle_debug
     @derives      [$_derive:ident, $($derives:ident,)*]
     @type         [$type:ident]
     @debug_format [$debug_format:tt]) => (
        newtype_index!(
            @handle_debug
            @derives      [$($derives,)*]
            @type         [$type]
            @debug_format [$debug_format]);
    );

    // Handle the case where someone wants to make the internal field public
    (@type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   pub idx
                   $($tokens:tt)*) => (
        newtype_index!(
            @pub          [pub]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // The default case is that the internal field is private
    (@type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   $($tokens:tt)*) => (
        newtype_index!(
            @pub          []
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // Append comma to end of derives list if it's missing
    (@pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   derive [$($derives:ident),*]
                   $($tokens:tt)*) => (
        newtype_index!(
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          derive [$($derives,)*]
                          $($tokens)*);
    );

    // By not including the @derives marker in this list nor in the default args, we can force it
    // to come first if it exists. When encodable is custom, just use the derives list as-is.
    (@pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   derive [$($derives:ident,)+]
                   ENCODABLE = custom
                   $($tokens:tt)*) => (
        newtype_index!(
            @derives      [$($derives,)+]
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // By not including the @derives marker in this list nor in the default args, we can force it
    // to come first if it exists. When encodable isn't custom, add serialization traits by default.
    (@pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   derive [$($derives:ident,)+]
                   $($tokens:tt)*) => (
        newtype_index!(
            @derives      [$($derives,)+ RustcDecodable, RustcEncodable,]
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // The case where no derives are added, but encodable is overridden. Don't
    // derive serialization traits
    (@pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   ENCODABLE = custom
                   $($tokens:tt)*) => (
        newtype_index!(
            @derives      []
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // The case where no derives are added, add serialization derives by default
    (@pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   $($tokens:tt)*) => (
        newtype_index!(
            @derives      [RustcDecodable, RustcEncodable,]
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // Rewrite final without comma to one that includes comma
    (@derives      [$($derives:ident,)*]
     @pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   $name:ident = $constant:expr) => (
        newtype_index!(
            @derives      [$($derives,)*]
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $name = $constant,);
    );

    // Rewrite final const without comma to one that includes comma
    (@derives      [$($derives:ident,)*]
     @pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$_max:expr]
     @debug_format [$debug_format:tt]
                   $(#[doc = $doc:expr])*
                   const $name:ident = $constant:expr) => (
        newtype_index!(
            @derives      [$($derives,)*]
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $(#[doc = $doc])* const $name = $constant,);
    );

    // Replace existing default for max
    (@derives      [$($derives:ident,)*]
     @pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$_max:expr]
     @debug_format [$debug_format:tt]
                   MAX = $max:expr,
                   $($tokens:tt)*) => (
        newtype_index!(
            @derives      [$($derives,)*]
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // Replace existing default for debug_format
    (@derives      [$($derives:ident,)*]
     @pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$_debug_format:tt]
                   DEBUG_FORMAT = $debug_format:tt,
                   $($tokens:tt)*) => (
        newtype_index!(
            @derives      [$($derives,)*]
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // Assign a user-defined constant
    (@derives      [$($derives:ident,)*]
     @pub          [$($pub:tt)*]
     @type         [$type:ident]
     @max          [$max:expr]
     @debug_format [$debug_format:tt]
                   $(#[doc = $doc:expr])*
                   const $name:ident = $constant:expr,
                   $($tokens:tt)*) => (
        $(#[doc = $doc])*
        pub const $name: $type = $type($constant);
        newtype_index!(
            @derives      [$($derives,)*]
            @pub          [$($pub)*]
            @type         [$type]
            @max          [$max]
            @debug_format [$debug_format]
                          $($tokens)*);
    );
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IndexVec<I: Idx, T> {
    pub raw: Vec<T>,
    _marker: PhantomData<fn(&I)>
}

// Whether `IndexVec` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, T> Send for IndexVec<I, T> where T: Send {}

impl<I: Idx, T: serialize::Encodable> serialize::Encodable for IndexVec<I, T> {
    fn encode<S: serialize::Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        serialize::Encodable::encode(&self.raw, s)
    }
}

impl<I: Idx, T: serialize::Decodable> serialize::Decodable for IndexVec<I, T> {
    fn decode<D: serialize::Decoder>(d: &mut D) -> Result<Self, D::Error> {
        serialize::Decodable::decode(d).map(|v| {
            IndexVec { raw: v, _marker: PhantomData }
        })
    }
}

impl<I: Idx, T: fmt::Debug> fmt::Debug for IndexVec<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

pub type Enumerated<I, J> = iter::Map<iter::Enumerate<J>, IntoIdx<I>>;

impl<I: Idx, T> IndexVec<I, T> {
    #[inline]
    pub fn new() -> Self {
        IndexVec { raw: Vec::new(), _marker: PhantomData }
    }

    #[inline]
    pub fn from_raw(raw: Vec<T>) -> Self {
        IndexVec { raw, _marker: PhantomData }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        IndexVec { raw: Vec::with_capacity(capacity), _marker: PhantomData }
    }

    #[inline]
    pub fn from_elem<S>(elem: T, universe: &IndexVec<I, S>) -> Self
        where T: Clone
    {
        IndexVec { raw: vec![elem; universe.len()], _marker: PhantomData }
    }

    #[inline]
    pub fn from_elem_n(elem: T, n: usize) -> Self
        where T: Clone
    {
        IndexVec { raw: vec![elem; n], _marker: PhantomData }
    }

    #[inline]
    pub fn push(&mut self, d: T) -> I {
        let idx = I::new(self.len());
        self.raw.push(d);
        idx
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.raw.pop()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[inline]
    pub fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }

    #[inline]
    pub fn into_iter_enumerated(self) -> Enumerated<I, vec::IntoIter<T>>
    {
        self.raw.into_iter().enumerate().map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.raw.iter()
    }

    #[inline]
    pub fn iter_enumerated(&self) -> Enumerated<I, slice::Iter<'_, T>>
    {
        self.raw.iter().enumerate().map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn indices(&self) -> iter::Map<Range<usize>, IntoIdx<I>> {
        (0..self.len()).map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.raw.iter_mut()
    }

    #[inline]
    pub fn iter_enumerated_mut(&mut self) -> Enumerated<I, slice::IterMut<'_, T>>
    {
        self.raw.iter_mut().enumerate().map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn drain<'a, R: RangeBounds<usize>>(
        &'a mut self, range: R) -> impl Iterator<Item=T> + 'a {
        self.raw.drain(range)
    }

    #[inline]
    pub fn drain_enumerated<'a, R: RangeBounds<usize>>(
        &'a mut self, range: R) -> impl Iterator<Item=(I, T)> + 'a {
        self.raw.drain(range).enumerate().map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn last(&self) -> Option<I> {
        self.len().checked_sub(1).map(I::new)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.raw.shrink_to_fit()
    }

    #[inline]
    pub fn swap(&mut self, a: I, b: I) {
        self.raw.swap(a.index(), b.index())
    }

    #[inline]
    pub fn truncate(&mut self, a: usize) {
        self.raw.truncate(a)
    }

    #[inline]
    pub fn get(&self, index: I) -> Option<&T> {
        self.raw.get(index.index())
    }

    #[inline]
    pub fn get_mut(&mut self, index: I) -> Option<&mut T> {
        self.raw.get_mut(index.index())
    }

    /// Return mutable references to two distinct elements, a and b. Panics if a == b.
    #[inline]
    pub fn pick2_mut(&mut self, a: I, b: I) -> (&mut T, &mut T) {
        let (ai, bi) = (a.index(), b.index());
        assert!(ai != bi);

        if ai < bi {
            let (c1, c2) = self.raw.split_at_mut(bi);
            (&mut c1[ai], &mut c2[0])
        } else {
            let (c2, c1) = self.pick2_mut(b, a);
            (c1, c2)
        }
    }

    pub fn convert_index_type<Ix: Idx>(self) -> IndexVec<Ix, T> {
        IndexVec {
            raw: self.raw,
            _marker: PhantomData,
        }
    }
}

impl<I: Idx, T: Clone> IndexVec<I, T> {
    /// Grows the index vector so that it contains an entry for
    /// `elem`; if that is already true, then has no
    /// effect. Otherwise, inserts new values as needed by invoking
    /// `fill_value`.
    #[inline]
    pub fn ensure_contains_elem(&mut self, elem: I, fill_value: impl FnMut() -> T) {
        let min_new_len = elem.index() + 1;
        if self.len() < min_new_len {
            self.raw.resize_with(min_new_len, fill_value);
        }
    }

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        self.raw.resize(new_len, value)
    }

    #[inline]
    pub fn resize_to_elem(&mut self, elem: I, fill_value: impl FnMut() -> T) {
        let min_new_len = elem.index() + 1;
        self.raw.resize_with(min_new_len, fill_value);
    }
}

impl<I: Idx, T: Ord> IndexVec<I, T> {
    #[inline]
    pub fn binary_search(&self, value: &T) -> Result<I, I> {
        match self.raw.binary_search(value) {
            Ok(i) => Ok(Idx::new(i)),
            Err(i) => Err(Idx::new(i)),
        }
    }
}

impl<I: Idx, T> Index<I> for IndexVec<I, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: I) -> &T {
        &self.raw[index.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for IndexVec<I, T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut T {
        &mut self.raw[index.index()]
    }
}

impl<I: Idx, T> Default for IndexVec<I, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T> Extend<T> for IndexVec<I, T> {
    #[inline]
    fn extend<J: IntoIterator<Item = T>>(&mut self, iter: J) {
        self.raw.extend(iter);
    }
}

impl<I: Idx, T> FromIterator<T> for IndexVec<I, T> {
    #[inline]
    fn from_iter<J>(iter: J) -> Self where J: IntoIterator<Item=T> {
        IndexVec { raw: FromIterator::from_iter(iter), _marker: PhantomData }
    }
}

impl<I: Idx, T> IntoIterator for IndexVec<I, T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }

}

impl<'a, I: Idx, T> IntoIterator for &'a IndexVec<I, T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.raw.iter()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a mut IndexVec<I, T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.raw.iter_mut()
    }
}

pub struct IntoIdx<I: Idx> { _marker: PhantomData<fn(&I)> }
impl<I: Idx, T> FnOnce<((usize, T),)> for IntoIdx<I> {
    type Output = (I, T);

    extern "rust-call" fn call_once(self, ((n, t),): ((usize, T),)) -> Self::Output {
        (I::new(n), t)
    }
}

impl<I: Idx, T> FnMut<((usize, T),)> for IntoIdx<I> {
    extern "rust-call" fn call_mut(&mut self, ((n, t),): ((usize, T),)) -> Self::Output {
        (I::new(n), t)
    }
}

impl<I: Idx> FnOnce<(usize,)> for IntoIdx<I> {
    type Output = I;

    extern "rust-call" fn call_once(self, (n,): (usize,)) -> Self::Output {
        I::new(n)
    }
}

impl<I: Idx> FnMut<(usize,)> for IntoIdx<I> {
    extern "rust-call" fn call_mut(&mut self, (n,): (usize,)) -> Self::Output {
        I::new(n)
    }
}
