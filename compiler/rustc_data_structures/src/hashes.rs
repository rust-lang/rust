use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::fmt;
use std::ops::BitXorAssign;
use crate::stable_hasher::{StableHasher, StableHasherResult};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Hash64 {
    inner: u64,
}

impl Hash64 {
    pub const ZERO: Hash64 = Hash64 { inner: 0 };

    #[inline]
    pub(crate) fn new(n: u64) -> Self {
        Self { inner: n }
    }

    #[inline]
    pub fn as_u64(self) -> u64 {
        self.inner
    }
}

impl BitXorAssign<u64> for Hash64 {
    fn bitxor_assign(&mut self, rhs: u64) {
        self.inner ^= rhs;
    }
}

impl<S: Encoder> Encodable<S> for Hash64 {
    #[inline]
    fn encode(&self, s: &mut S) {
        s.emit_raw_bytes(&self.inner.to_le_bytes());
    }
}

impl<D: Decoder> Decodable<D> for Hash64 {
    #[inline]
    fn decode(d: &mut D) -> Self {
        Self { inner: u64::from_le_bytes(d.read_raw_bytes(8).try_into().unwrap()) }
    }
}

impl StableHasherResult for Hash64 {
    #[inline]
    fn finish(hasher: StableHasher) -> Self {
        Self { inner: hasher.finalize().0 }
    }
}

impl fmt::Debug for Hash64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl fmt::LowerHex for Hash64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.inner, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Hash128 {
    inner: u128,
}

impl Hash128 {
    #[inline]
    pub fn truncate(self) -> Hash64 {
        Hash64 { inner: self.inner as u64 }
    }

    #[inline]
    pub fn wrapping_add(self, other: Self) -> Self {
        Self {
            inner: self.inner.wrapping_add(other.inner),
        }
    }

    #[inline]
    pub fn as_u128(self) -> u128 {
        self.inner
    }
}

impl<S: Encoder> Encodable<S> for Hash128 {
    #[inline]
    fn encode(&self, s: &mut S) {
        s.emit_raw_bytes(&self.inner.to_le_bytes());
    }
}

impl<D: Decoder> Decodable<D> for Hash128 {
    #[inline]
    fn decode(d: &mut D) -> Self {
        Self { inner: u128::from_le_bytes(d.read_raw_bytes(16).try_into().unwrap()) }
    }
}

impl StableHasherResult for Hash128 {
    #[inline]
    fn finish(hasher: StableHasher) -> Self {
        let (_0, _1) = hasher.finalize();
        Self { inner: u128::from(_0) | (u128::from(_1) << 64) }
    }
}

impl fmt::Debug for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl fmt::LowerHex for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.inner, f)
    }
}
