//! Calculation and management of a Strict Version Hash for crates
//!
//! The SVH is used for incremental compilation to track when HIR
//! nodes have changed between compilations, and also to detect
//! mismatches where we have two versions of the same crate that were
//! compiled from distinct sources.

use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::stable_hasher;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Svh {
    hash: u64,
}

impl Svh {
    /// Creates a new `Svh` given the hash. If you actually want to
    /// compute the SVH from some HIR, you want the `calculate_svh`
    /// function found in `rustc_incremental`.
    pub fn new(hash: u64) -> Svh {
        Svh { hash }
    }

    pub fn as_u64(&self) -> u64 {
        self.hash
    }

    pub fn to_string(&self) -> String {
        format!("{:016x}", self.hash)
    }
}

impl Hash for Svh {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.hash.to_le().hash(state);
    }
}

impl fmt::Display for Svh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.to_string())
    }
}

impl<S: Encoder> Encodable<S> for Svh {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u64(self.as_u64().to_le())
    }
}

impl<D: Decoder> Decodable<D> for Svh {
    fn decode(d: &mut D) -> Svh {
        Svh::new(u64::from_le(d.read_u64()))
    }
}

impl<T> stable_hasher::HashStable<T> for Svh {
    #[inline]
    fn hash_stable(&self, ctx: &mut T, hasher: &mut stable_hasher::StableHasher) {
        let Svh { hash } = *self;
        hash.hash_stable(ctx, hasher);
    }
}
