use std::cmp::Ordering;
use std::fmt;

use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use crate::stable_hasher::{HashStable, StableHasher};

/// A packed 128-bit integer. Useful for reducing the size of structures in
/// some cases.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(packed(8))]
pub struct Pu128(pub u128);

impl Pu128 {
    #[inline]
    pub fn get(self) -> u128 {
        self.0
    }
}

impl From<Pu128> for u128 {
    #[inline]
    fn from(value: Pu128) -> Self {
        value.get()
    }
}

impl From<u128> for Pu128 {
    #[inline]
    fn from(value: u128) -> Self {
        Self(value)
    }
}

impl PartialEq<u128> for Pu128 {
    #[inline]
    fn eq(&self, other: &u128) -> bool {
        ({ self.0 }) == *other
    }
}

impl PartialOrd<u128> for Pu128 {
    #[inline]
    fn partial_cmp(&self, other: &u128) -> Option<Ordering> {
        { self.0 }.partial_cmp(other)
    }
}

impl fmt::Display for Pu128 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        { self.0 }.fmt(f)
    }
}

impl fmt::UpperHex for Pu128 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        { self.0 }.fmt(f)
    }
}

impl<CTX> HashStable<CTX> for Pu128 {
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        { self.0 }.hash_stable(ctx, hasher)
    }
}

impl<S: Encoder> Encodable<S> for Pu128 {
    #[inline]
    fn encode(&self, s: &mut S) {
        { self.0 }.encode(s);
    }
}

impl<D: Decoder> Decodable<D> for Pu128 {
    #[inline]
    fn decode(d: &mut D) -> Self {
        Self(u128::decode(d))
    }
}
