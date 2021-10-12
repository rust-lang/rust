use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, RangeBounds};
use std::slice;
use std::vec;

/// Represents some newtyped `usize` wrapper.
///
/// Purpose: avoid mixing indexes for different bitvector domains.
pub trait Idx: Copy + 'static + Ord + Debug + Hash {
    fn new(idx: usize) -> Self;

    fn index(self) -> usize;

    fn increment_by(&mut self, amount: usize) {
        *self = self.plus(amount);
    }

    fn plus(self, amount: usize) -> Self {
        Self::new(self.index() + amount)
    }
}

impl Idx for usize {
    #[inline]
    fn new(idx: usize) -> Self {
        idx
    }
    #[inline]
    fn index(self) -> usize {
        self
    }
}

impl Idx for u32 {
    #[inline]
    fn new(idx: usize) -> Self {
        assert!(idx <= u32::MAX as usize);
        idx as u32
    }
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

/// Creates a struct type `S` that can be used as an index with
/// `IndexVec` and so on.
///
/// There are two ways of interacting with these indices:
///
/// - The `From` impls are the preferred way. So you can do
///   `S::from(v)` with a `usize` or `u32`. And you can convert back
///   to an integer with `u32::from(s)`.
///
/// - Alternatively, you can use the methods `S::new(v)` and `s.index()`
///   to create/return a value.
///
/// Internally, the index uses a u32, so the index must not exceed
/// `u32::MAX`. You can also customize things like the `Debug` impl,
/// what traits are derived, and so forth via the macro.
#[macro_export]
#[allow_internal_unstable(step_trait, rustc_attrs, trusted_step)]
macro_rules! newtype_index {
    // ---- public rules ----

    // Use default constants
    ($(#[$attrs:meta])* $v:vis struct $name:ident { .. }) => (
        $crate::newtype_index!(
            // Leave out derives marker so we can use its absence to ensure it comes first
            @attrs        [$(#[$attrs])*]
            @type         [$name]
            // shave off 256 indices at the end to allow space for packing these indices into enums
            @max          [0xFFFF_FF00]
            @vis          [$v]
            @debug_format ["{}"]);
    );

    // Define any constants
    ($(#[$attrs:meta])* $v:vis struct $name:ident { $($tokens:tt)+ }) => (
        $crate::newtype_index!(
            // Leave out derives marker so we can use its absence to ensure it comes first
            @attrs        [$(#[$attrs])*]
            @type         [$name]
            // shave off 256 indices at the end to allow space for packing these indices into enums
            @max          [0xFFFF_FF00]
            @vis          [$v]
            @debug_format ["{}"]
                          $($tokens)+);
    );

    // ---- private rules ----

    // Base case, user-defined constants (if any) have already been defined
    (@derives      [$($derives:ident,)*]
     @attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]) => (
        $(#[$attrs])*
        #[derive(Copy, PartialEq, Eq, Hash, PartialOrd, Ord, $($derives),*)]
        #[rustc_layout_scalar_valid_range_end($max)]
        $v struct $type {
            private: u32
        }

        impl Clone for $type {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl $type {
            $v const MAX_AS_U32: u32 = $max;

            $v const MAX: Self = Self::from_u32($max);

            #[inline]
            $v const fn from_usize(value: usize) -> Self {
                #[cfg(not(bootstrap))]
                assert!(value <= ($max as usize));
                #[cfg(bootstrap)]
                [()][(value > ($max as usize)) as usize];
                unsafe {
                    Self::from_u32_unchecked(value as u32)
                }
            }

            #[inline]
            $v const fn from_u32(value: u32) -> Self {
                #[cfg(not(bootstrap))]
                assert!(value <= $max);
                #[cfg(bootstrap)]
                [()][(value > $max) as usize];
                unsafe {
                    Self::from_u32_unchecked(value)
                }
            }

            #[inline]
            $v const unsafe fn from_u32_unchecked(value: u32) -> Self {
                Self { private: value }
            }

            /// Extracts the value of this index as an integer.
            #[inline]
            $v const fn index(self) -> usize {
                self.as_usize()
            }

            /// Extracts the value of this index as a `u32`.
            #[inline]
            $v const fn as_u32(self) -> u32 {
                self.private
            }

            /// Extracts the value of this index as a `usize`.
            #[inline]
            $v const fn as_usize(self) -> usize {
                self.as_u32() as usize
            }
        }

        impl std::ops::Add<usize> for $type {
            type Output = Self;

            fn add(self, other: usize) -> Self {
                Self::from_usize(self.index() + other)
            }
        }

        impl $crate::vec::Idx for $type {
            #[inline]
            fn new(value: usize) -> Self {
                Self::from_usize(value)
            }

            #[inline]
            fn index(self) -> usize {
                self.as_usize()
            }
        }

        impl ::std::iter::Step for $type {
            #[inline]
            fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                <usize as ::std::iter::Step>::steps_between(
                    &Self::index(*start),
                    &Self::index(*end),
                )
            }

            #[inline]
            fn forward_checked(start: Self, u: usize) -> Option<Self> {
                Self::index(start).checked_add(u).map(Self::from_usize)
            }

            #[inline]
            fn backward_checked(start: Self, u: usize) -> Option<Self> {
                Self::index(start).checked_sub(u).map(Self::from_usize)
            }
        }

        // Safety: The implementation of `Step` upholds all invariants.
        unsafe impl ::std::iter::TrustedStep for $type {}

        impl From<$type> for u32 {
            #[inline]
            fn from(v: $type) -> u32 {
                v.as_u32()
            }
        }

        impl From<$type> for usize {
            #[inline]
            fn from(v: $type) -> usize {
                v.as_usize()
            }
        }

        impl From<usize> for $type {
            #[inline]
            fn from(value: usize) -> Self {
                Self::from_usize(value)
            }
        }

        impl From<u32> for $type {
            #[inline]
            fn from(value: u32) -> Self {
                Self::from_u32(value)
            }
        }

        $crate::newtype_index!(
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
            fn fmt(&self, fmt: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(fmt, $debug_format, self.as_u32())
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
        $crate::newtype_index!(
            @handle_debug
            @derives      [$($derives,)*]
            @type         [$type]
            @debug_format [$debug_format]);
    );

    // Append comma to end of derives list if it's missing
    (@attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   derive [$($derives:ident),*]
                   $($tokens:tt)*) => (
        $crate::newtype_index!(
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          derive [$($derives,)*]
                          $($tokens)*);
    );

    // By not including the @derives marker in this list nor in the default args, we can force it
    // to come first if it exists. When encodable is custom, just use the derives list as-is.
    (@attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   derive [$($derives:ident,)+]
                   ENCODABLE = custom
                   $($tokens:tt)*) => (
        $crate::newtype_index!(
            @attrs        [$(#[$attrs])*]
            @derives      [$($derives,)+]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // By not including the @derives marker in this list nor in the default args, we can force it
    // to come first if it exists. When encodable isn't custom, add serialization traits by default.
    (@attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   derive [$($derives:ident,)+]
                   $($tokens:tt)*) => (
        $crate::newtype_index!(
            @derives      [$($derives,)+]
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $($tokens)*);
        $crate::newtype_index!(@serializable $type);
    );

    // The case where no derives are added, but encodable is overridden. Don't
    // derive serialization traits
    (@attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   ENCODABLE = custom
                   $($tokens:tt)*) => (
        $crate::newtype_index!(
            @derives      []
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // The case where no derives are added, add serialization derives by default
    (@attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   $($tokens:tt)*) => (
        $crate::newtype_index!(
            @derives      []
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $($tokens)*);
        $crate::newtype_index!(@serializable $type);
    );

    (@serializable $type:ident) => (
        impl<D: ::rustc_serialize::Decoder> ::rustc_serialize::Decodable<D> for $type {
            fn decode(d: &mut D) -> Result<Self, D::Error> {
                d.read_u32().map(Self::from_u32)
            }
        }
        impl<E: ::rustc_serialize::Encoder> ::rustc_serialize::Encodable<E> for $type {
            fn encode(&self, e: &mut E) -> Result<(), E::Error> {
                e.emit_u32(self.private)
            }
        }
    );

    // Rewrite final without comma to one that includes comma
    (@derives      [$($derives:ident,)*]
     @attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   $name:ident = $constant:expr) => (
        $crate::newtype_index!(
            @derives      [$($derives,)*]
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $name = $constant,);
    );

    // Rewrite final const without comma to one that includes comma
    (@derives      [$($derives:ident,)*]
     @attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   $(#[doc = $doc:expr])*
                   const $name:ident = $constant:expr) => (
        $crate::newtype_index!(
            @derives      [$($derives,)*]
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $(#[doc = $doc])* const $name = $constant,);
    );

    // Replace existing default for max
    (@derives      [$($derives:ident,)*]
     @attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$_max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   MAX = $max:expr,
                   $($tokens:tt)*) => (
        $crate::newtype_index!(
            @derives      [$($derives,)*]
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // Replace existing default for debug_format
    (@derives      [$($derives:ident,)*]
     @attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$_debug_format:tt]
                   DEBUG_FORMAT = $debug_format:tt,
                   $($tokens:tt)*) => (
        $crate::newtype_index!(
            @derives      [$($derives,)*]
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $($tokens)*);
    );

    // Assign a user-defined constant
    (@derives      [$($derives:ident,)*]
     @attrs        [$(#[$attrs:meta])*]
     @type         [$type:ident]
     @max          [$max:expr]
     @vis          [$v:vis]
     @debug_format [$debug_format:tt]
                   $(#[doc = $doc:expr])*
                   const $name:ident = $constant:expr,
                   $($tokens:tt)*) => (
        $(#[doc = $doc])*
        $v const $name: $type = $type::from_u32($constant);
        $crate::newtype_index!(
            @derives      [$($derives,)*]
            @attrs        [$(#[$attrs])*]
            @type         [$type]
            @max          [$max]
            @vis          [$v]
            @debug_format [$debug_format]
                          $($tokens)*);
    );
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IndexVec<I: Idx, T> {
    pub raw: Vec<T>,
    _marker: PhantomData<fn(&I)>,
}

// Whether `IndexVec` is `Send` depends only on the data,
// not the phantom data.
unsafe impl<I: Idx, T> Send for IndexVec<I, T> where T: Send {}

impl<S: Encoder, I: Idx, T: Encodable<S>> Encodable<S> for IndexVec<I, T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        Encodable::encode(&self.raw, s)
    }
}

impl<S: Encoder, I: Idx, T: Encodable<S>> Encodable<S> for &IndexVec<I, T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        Encodable::encode(&self.raw, s)
    }
}

impl<D: Decoder, I: Idx, T: Decodable<D>> Decodable<D> for IndexVec<I, T> {
    fn decode(d: &mut D) -> Result<Self, D::Error> {
        Decodable::decode(d).map(|v| IndexVec { raw: v, _marker: PhantomData })
    }
}

impl<I: Idx, T: fmt::Debug> fmt::Debug for IndexVec<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

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
    where
        T: Clone,
    {
        IndexVec { raw: vec![elem; universe.len()], _marker: PhantomData }
    }

    #[inline]
    pub fn from_elem_n(elem: T, n: usize) -> Self
    where
        T: Clone,
    {
        IndexVec { raw: vec![elem; n], _marker: PhantomData }
    }

    /// Create an `IndexVec` with `n` elements, where the value of each
    /// element is the result of `func(i)`. (The underlying vector will
    /// be allocated only once, with a capacity of at least `n`.)
    #[inline]
    pub fn from_fn_n(func: impl FnMut(I) -> T, n: usize) -> Self {
        let indices = (0..n).map(I::new);
        Self::from_raw(indices.map(func).collect())
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

    /// Gives the next index that will be assigned when `push` is
    /// called.
    #[inline]
    pub fn next_index(&self) -> I {
        I::new(self.len())
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
    pub fn into_iter_enumerated(
        self,
    ) -> impl DoubleEndedIterator<Item = (I, T)> + ExactSizeIterator {
        self.raw.into_iter().enumerate().map(|(n, t)| (I::new(n), t))
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.raw.iter()
    }

    #[inline]
    pub fn iter_enumerated(
        &self,
    ) -> impl DoubleEndedIterator<Item = (I, &T)> + ExactSizeIterator + '_ {
        self.raw.iter().enumerate().map(|(n, t)| (I::new(n), t))
    }

    #[inline]
    pub fn indices(&self) -> impl DoubleEndedIterator<Item = I> + ExactSizeIterator + 'static {
        (0..self.len()).map(|n| I::new(n))
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.raw.iter_mut()
    }

    #[inline]
    pub fn iter_enumerated_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (I, &mut T)> + ExactSizeIterator + '_ {
        self.raw.iter_mut().enumerate().map(|(n, t)| (I::new(n), t))
    }

    #[inline]
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> impl Iterator<Item = T> + '_ {
        self.raw.drain(range)
    }

    #[inline]
    pub fn drain_enumerated<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> impl Iterator<Item = (I, T)> + '_ {
        self.raw.drain(range).enumerate().map(|(n, t)| (I::new(n), t))
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

    /// Returns mutable references to two distinct elements, a and b. Panics if a == b.
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

    /// Returns mutable references to three distinct elements or panics otherwise.
    #[inline]
    pub fn pick3_mut(&mut self, a: I, b: I, c: I) -> (&mut T, &mut T, &mut T) {
        let (ai, bi, ci) = (a.index(), b.index(), c.index());
        assert!(ai != bi && bi != ci && ci != ai);
        let len = self.raw.len();
        assert!(ai < len && bi < len && ci < len);
        let ptr = self.raw.as_mut_ptr();
        unsafe { (&mut *ptr.add(ai), &mut *ptr.add(bi), &mut *ptr.add(ci)) }
    }

    pub fn convert_index_type<Ix: Idx>(self) -> IndexVec<Ix, T> {
        IndexVec { raw: self.raw, _marker: PhantomData }
    }

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
    pub fn resize_to_elem(&mut self, elem: I, fill_value: impl FnMut() -> T) {
        let min_new_len = elem.index() + 1;
        self.raw.resize_with(min_new_len, fill_value);
    }
}

/// `IndexVec` is often used as a map, so it provides some map-like APIs.
impl<I: Idx, T> IndexVec<I, Option<T>> {
    #[inline]
    pub fn insert(&mut self, index: I, value: T) -> Option<T> {
        self.ensure_contains_elem(index, || None);
        self[index].replace(value)
    }

    #[inline]
    pub fn get_or_insert_with(&mut self, index: I, value: impl FnOnce() -> T) -> &mut T {
        self.ensure_contains_elem(index, || None);
        self[index].get_or_insert_with(value)
    }

    #[inline]
    pub fn remove(&mut self, index: I) -> Option<T> {
        self.ensure_contains_elem(index, || None);
        self[index].take()
    }
}

impl<I: Idx, T: Clone> IndexVec<I, T> {
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        self.raw.resize(new_len, value)
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

    #[inline]
    fn extend_one(&mut self, item: T) {
        self.raw.push(item);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        self.raw.reserve(additional);
    }
}

impl<I: Idx, T> FromIterator<T> for IndexVec<I, T> {
    #[inline]
    fn from_iter<J>(iter: J) -> Self
    where
        J: IntoIterator<Item = T>,
    {
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

#[cfg(test)]
mod tests;
