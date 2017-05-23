// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_serialize::{Encodable, Decodable, Encoder, Decoder};
use rustc_data_structures::stable_hasher;
use std::mem;
use std::slice;

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy)]
pub struct Fingerprint(u64, u64);

impl Fingerprint {
    #[inline]
    pub fn zero() -> Fingerprint {
        Fingerprint(0, 0)
    }

    #[inline]
    pub fn from_smaller_hash(hash: u64) -> Fingerprint {
        Fingerprint(hash, hash)
    }

    #[inline]
    pub fn to_smaller_hash(&self) -> u64 {
        self.0
    }

    pub fn to_hex(&self) -> String {
        format!("{:x}{:x}", self.0, self.1)
    }
}

impl Encodable for Fingerprint {
    #[inline]
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u64(self.0.to_le())?;
        s.emit_u64(self.1.to_le())
    }
}

impl Decodable for Fingerprint {
    #[inline]
    fn decode<D: Decoder>(d: &mut D) -> Result<Fingerprint, D::Error> {
        let _0 = u64::from_le(d.read_u64()?);
        let _1 = u64::from_le(d.read_u64()?);
        Ok(Fingerprint(_0, _1))
    }
}

impl ::std::fmt::Display for Fingerprint {
    fn fmt(&self, formatter: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(formatter, "{:x}-{:x}", self.0, self.1)
    }
}

impl stable_hasher::StableHasherResult for Fingerprint {
    fn finish(mut hasher: stable_hasher::StableHasher<Self>) -> Self {
        let hash_bytes: &[u8] = hasher.finalize();

        assert!(hash_bytes.len() >= mem::size_of::<u64>() * 2);
        let hash_bytes: &[u64] = unsafe {
            slice::from_raw_parts(hash_bytes.as_ptr() as *const u64, 2)
        };

        // The bytes returned bytes the Blake2B hasher are always little-endian.
        Fingerprint(u64::from_le(hash_bytes[0]), u64::from_le(hash_bytes[1]))
    }
}

impl<CTX> stable_hasher::HashStable<CTX> for Fingerprint {
    #[inline]
    fn hash_stable<W: stable_hasher::StableHasherResult>(&self,
                                          _: &mut CTX,
                                          hasher: &mut stable_hasher::StableHasher<W>) {
        ::std::hash::Hash::hash(self, hasher);
    }
}
