//! Calculation and management of a Strict Version Hash for crates
//!
//! The SVH is used for incremental compilation to track when HIR
//! nodes have changed between compilations, and also to detect
//! mismatches where we have two versions of the same crate that were
//! compiled from distinct sources.

use crate::fingerprint::Fingerprint;
use std::fmt;

use crate::stable_hasher;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Encodable, Decodable, Hash)]
pub struct Svh {
    hash: Fingerprint,
}

impl Svh {
    /// Creates a new `Svh` given the hash. If you actually want to
    /// compute the SVH from some HIR, you want the `calculate_svh`
    /// function found in `rustc_incremental`.
    pub fn new(hash: Fingerprint) -> Svh {
        Svh { hash }
    }

    pub fn as_u64(&self) -> u64 {
        self.hash.to_smaller_hash().as_u64()
    }

    pub fn to_string(&self) -> String {
        format!("{:016x}", self.hash.to_smaller_hash())
    }
}

impl fmt::Display for Svh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.to_string())
    }
}

impl<T> stable_hasher::HashStable<T> for Svh {
    #[inline]
    fn hash_stable(&self, ctx: &mut T, hasher: &mut stable_hasher::StableHasher) {
        let Svh { hash } = *self;
        hash.hash_stable(ctx, hasher);
    }
}
