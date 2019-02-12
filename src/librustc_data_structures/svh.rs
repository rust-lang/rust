//! Calculation and management of a Strict Version Hash for crates
//!
//! The SVH is used for incremental compilation to track when HIR
//! nodes have changed between compilations, and also to detect
//! mismatches where we have two versions of the same crate that were
//! compiled from distinct sources.

use std::fmt;
use std::hash::{Hash, Hasher};
use serialize::{Encodable, Decodable, Encoder, Decoder};

use crate::stable_hasher;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Svh {
    hash: u64,
}

impl Svh {
    /// Creates a new `Svh` given the hash. If you actually want to
    /// compute the SVH from some HIR, you want the `calculate_svh`
    /// function found in `librustc_incremental`.
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
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        self.hash.to_le().hash(state);
    }
}

impl fmt::Display for Svh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.to_string())
    }
}

impl Encodable for Svh {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u64(self.as_u64().to_le())
    }
}

impl Decodable for Svh {
    fn decode<D: Decoder>(d: &mut D) -> Result<Svh, D::Error> {
        d.read_u64()
         .map(u64::from_le)
         .map(Svh::new)
    }
}

impl<T> stable_hasher::HashStable<T> for Svh {
    #[inline]
    fn hash_stable<W: stable_hasher::StableHasherResult>(
        &self,
        ctx: &mut T,
        hasher: &mut stable_hasher::StableHasher<W>
    ) {
        let Svh {
            hash
        } = *self;
        hash.hash_stable(ctx, hasher);
    }
}
