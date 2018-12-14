use std::fmt::{self, Display, Debug};
use std::ops::Deref;
use std::{slice, vec};
use crate::arena::Arena;

use serialize::{Encodable, Decodable, Encoder, Decoder};

use rustc_data_structures::stable_hasher::{StableHasher, StableHasherResult,
                                           HashStable};

pub trait IteratorExt: Iterator {
    fn collect_hir_vec<'a>(self, arena: &'a Arena<'a>) -> P<'a, [Self::Item]>;
}

impl<T: Iterator> IteratorExt for T where T::Item: Copy {
    fn collect_hir_vec<'a>(self, arena: &'a Arena<'a>) -> P<'a, [Self::Item]> {
        P::from_iter(arena, self)
    }
}

#[derive(Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct P<'a, T: ?Sized>(&'a T);

impl<'a, T: 'a+?Sized> Clone for P<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        P(self.0)
    }
}
impl<'a, T: 'a+?Sized> Copy for P<'a, T> {}

impl<'a, T: ?Sized> Deref for P<'a, T> {
    type Target = &'a T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T: Copy> P<'a, [T]> {
    #[inline]
    pub fn from_slice(arena: &'a Arena<'a>, slice: &[T]) -> Self {
        if slice.is_empty() {
            P::new()
        } else {
            P(arena.alloc_slice(slice))
        }
    }

    #[inline]
    pub fn from_iter<I: IntoIterator<Item=T>>(arena: &'a Arena<'a>, iter: I) -> Self {
        P(arena.alloc_from_iter(iter))
    }
}

impl<'a, T: Copy> P<'a, T> {
    /// Equivalent to and_then(|x| x)
    #[inline]
    pub fn into_inner(&self) -> T {
        *self.0
    }

    /// Move out of the pointer.
    /// Intended for chaining transformations not covered by `map`.
    #[inline]
    pub fn and_then<U, F>(&self, f: F) -> U where
        F: FnOnce(T) -> U,
    {
        f(*self.0)
    }

    #[inline]
    pub fn alloc(arena: &'a Arena<'a>, inner: T) -> Self {
        P(arena.alloc(inner))
    }
}

impl<'a, T> P<'a, T> {
    #[inline]
    pub fn empty_thin() -> P<'a, P<'a, [T]>> {
        P(&P(&[]))
    }
}

impl<'a, T: ?Sized> P<'a, T> {
    // FIXME: Doesn't work with deserialization
    #[inline]
    pub fn from_existing(val: &'a T) -> Self {
        P(val)
    }
}

impl<T: ?Sized + Debug> Debug for P<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self.0, f)
    }
}

impl<T: Display> Display for P<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T> fmt::Pointer for P<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&(self.0 as *const T), f)
    }
}

impl<T: Decodable> Decodable for P<'_, T> {
    fn decode<D: Decoder>(_d: &mut D) -> Result<Self, D::Error> {
        panic!()
    }
}

impl<T: Encodable> Encodable for P<'_, T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<'a, T> P<'a, [T]> {
    #[inline]
    pub fn new() -> Self {
        P(&[])
    }

    #[inline(never)]
    pub fn into_vec(self) -> Vec<T> where T: Clone {
        (*self.0).iter().cloned().collect()
    }
}

impl<T> Default for P<'_, [T]> {
    /// Creates an empty `P<[T]>`.
    #[inline]
    fn default() -> Self {
        P::new()
    }
}

impl<T: Clone> Into<Vec<T>> for P<'_, [T]> {
    fn into(self) -> Vec<T> {
        self.into_vec()
    }
}

impl<T: Clone> IntoIterator for P<'_, [T]> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a P<'_, [T]> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: Encodable> Encodable for P<'_, [T]> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        Encodable::encode(&**self, s)
    }
}

impl<T: Decodable> Decodable for P<'_, [T]> {
    fn decode<D: Decoder>(_d: &mut D) -> Result<Self, D::Error> {
        panic!()
    }
}

impl<CTX, T> HashStable<CTX> for P<'_, T>
    where T: ?Sized + HashStable<CTX>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (**self).hash_stable(hcx, hasher);
    }
}
