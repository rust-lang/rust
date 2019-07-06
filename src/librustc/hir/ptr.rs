// HACK(eddyb) this is a copy of `syntax::ptr`, minus the mutation (the HIR is
// frozen anyway). The only reason for doing this instead of replacing `P<T>`
// with `Box<T>` in HIR, is that `&Box<[T]>` doesn't implement `IntoIterator`.

use std::fmt::{self, Display, Debug};
use std::iter::FromIterator;
use std::ops::Deref;
use std::{slice, vec};

use serialize::{Encodable, Decodable, Encoder, Decoder};

use rustc_data_structures::stable_hasher::{StableHasher, StableHasherResult,
                                           HashStable};
/// An owned smart pointer.
#[derive(Hash, PartialEq, Eq)]
pub struct P<T: ?Sized> {
    ptr: Box<T>
}

/// Construct a `P<T>` from a `T` value.
#[allow(non_snake_case)]
pub fn P<T: 'static>(value: T) -> P<T> {
    P {
        ptr: box value
    }
}

impl<T: 'static> P<T> {
    // HACK(eddyb) used by HIR lowering in a few places still.
    // NOTE: do not make this more public than `pub(super)`.
    pub(super) fn into_inner(self) -> T {
        *self.ptr
    }
}

impl<T: ?Sized> Deref for P<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.ptr
    }
}

impl<T: ?Sized + Debug> Debug for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.ptr, f)
    }
}

impl<T: Display> Display for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T: 'static + Decodable> Decodable for P<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<P<T>, D::Error> {
        Decodable::decode(d).map(P)
    }
}

impl<T: Encodable> Encodable for P<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<T> P<[T]> {
    pub const fn new() -> P<[T]> {
        // HACK(eddyb) bypass the lack of a `const fn` to create an empty `Box<[T]>`
        // (as trait methods, `default` in this case, can't be `const fn` yet).
        P {
            ptr: unsafe {
                use std::ptr::NonNull;
                std::mem::transmute(NonNull::<[T; 0]>::dangling() as NonNull<[T]>)
            },
        }
    }

    #[inline(never)]
    pub fn from_vec(v: Vec<T>) -> P<[T]> {
        P { ptr: v.into_boxed_slice() }
    }

    // HACK(eddyb) used by HIR lowering in a few places still.
    // NOTE: do not make this more public than `pub(super)`,
    // and do not make this into an `IntoIterator` impl.
    pub(super) fn into_iter(self) -> vec::IntoIter<T> {
        self.ptr.into_vec().into_iter()
    }
}


impl<T> Default for P<[T]> {
    /// Creates an empty `P<[T]>`.
    fn default() -> P<[T]> {
        P::new()
    }
}

impl<T> From<Vec<T>> for P<[T]> {
    fn from(v: Vec<T>) -> Self {
        P::from_vec(v)
    }
}

impl<T> FromIterator<T> for P<[T]> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> P<[T]> {
        P::from_vec(iter.into_iter().collect())
    }
}

impl<'a, T> IntoIterator for &'a P<[T]> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.ptr.into_iter()
    }
}

impl<T: Encodable> Encodable for P<[T]> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        Encodable::encode(&**self, s)
    }
}

impl<T: Decodable> Decodable for P<[T]> {
    fn decode<D: Decoder>(d: &mut D) -> Result<P<[T]>, D::Error> {
        Ok(P::from_vec(Decodable::decode(d)?))
    }
}

impl<CTX, T> HashStable<CTX> for P<T>
    where T: ?Sized + HashStable<CTX>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (**self).hash_stable(hcx, hasher);
    }
}
