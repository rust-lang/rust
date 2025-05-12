//! Calculation and management of a Strict Version Hash for crates
//!
//! The SVH is used for incremental compilation to track when HIR
//! nodes have changed between compilations, and also to detect
//! mismatches where we have two versions of the same crate that were
//! compiled from distinct sources.

use std::fmt;

use rustc_macros::{Decodable_NoContext, Encodable_NoContext};

use crate::fingerprint::Fingerprint;
use crate::stable_hasher;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Encodable_NoContext, Decodable_NoContext, Hash)]
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

    pub fn as_u128(self) -> u128 {
        self.hash.as_u128()
    }

    pub fn to_hex(self) -> String {
        format!("{:032x}", self.hash.as_u128())
    }
}

impl fmt::Display for Svh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.to_hex())
    }
}

impl<T> stable_hasher::HashStable<T> for Svh {
    #[inline]
    fn hash_stable(&self, ctx: &mut T, hasher: &mut stable_hasher::StableHasher) {
        let Svh { hash } = *self;
        hash.hash_stable(ctx, hasher);
    }
}
