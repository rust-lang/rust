//! rustc encodes a lot of hashes. If hashes are stored as `u64` or `u128`, a `derive(Encodable)`
//! will apply varint encoding to the hashes, which is less efficient than directly encoding the 8
//! or 16 bytes of the hash.
//!
//! The types in this module represent 64-bit or 128-bit hashes produced by a `StableHasher`.
//! `Hash64` and `Hash128` expose some utility functions to encourage users to not extract the inner
//! hash value as an integer type and accidentally apply varint encoding to it.
//!
//! In contrast with `Fingerprint`, users of these types cannot and should not attempt to construct
//! and decompose these types into constituent pieces. The point of these types is only to
//! connect the fact that they can only be produced by a `StableHasher` to their
//! `Encode`/`Decode` impls.

use std::fmt;
use std::ops::BitXorAssign;

use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use crate::stable_hasher::{FromStableHash, StableHasherHash};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Hash64 {
    inner: u64,
}

impl Hash64 {
    pub const ZERO: Hash64 = Hash64 { inner: 0 };

    #[inline]
    pub fn new(n: u64) -> Self {
        Self { inner: n }
    }

    #[inline]
    pub fn as_u64(self) -> u64 {
        self.inner
    }
}

impl BitXorAssign<u64> for Hash64 {
    #[inline]
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

impl FromStableHash for Hash64 {
    type Hash = StableHasherHash;

    #[inline]
    fn from(StableHasherHash([_0, __1]): Self::Hash) -> Self {
        Self { inner: _0 }
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Hash128 {
    inner: u128,
}

// We expect Hash128 to be well mixed. So there's no point in hashing both parts.
//
// This also allows using Hash128-containing types in UnHash-based hashmaps, which would otherwise
// debug_assert! that we're hashing more than a single u64.
impl std::hash::Hash for Hash128 {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        h.write_u64(self.truncate().as_u64());
    }
}

impl Hash128 {
    #[inline]
    pub fn truncate(self) -> Hash64 {
        Hash64 { inner: self.inner as u64 }
    }

    #[inline]
    pub fn wrapping_add(self, other: Self) -> Self {
        Self { inner: self.inner.wrapping_add(other.inner) }
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

impl FromStableHash for Hash128 {
    type Hash = StableHasherHash;

    #[inline]
    fn from(StableHasherHash([_0, _1]): Self::Hash) -> Self {
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
