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
use rustc_data_structures::ToHex;

const FINGERPRINT_LENGTH: usize = 16;

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy)]
pub struct Fingerprint(pub [u8; FINGERPRINT_LENGTH]);

impl Fingerprint {
    #[inline]
    pub fn zero() -> Fingerprint {
        Fingerprint([0; FINGERPRINT_LENGTH])
    }

    pub fn from_smaller_hash(hash: u64) -> Fingerprint {
        let mut result = Fingerprint::zero();
        result.0[0] = (hash >>  0) as u8;
        result.0[1] = (hash >>  8) as u8;
        result.0[2] = (hash >> 16) as u8;
        result.0[3] = (hash >> 24) as u8;
        result.0[4] = (hash >> 32) as u8;
        result.0[5] = (hash >> 40) as u8;
        result.0[6] = (hash >> 48) as u8;
        result.0[7] = (hash >> 56) as u8;
        result
    }

    pub fn to_smaller_hash(&self) -> u64 {
        ((self.0[0] as u64) <<  0) |
        ((self.0[1] as u64) <<  8) |
        ((self.0[2] as u64) << 16) |
        ((self.0[3] as u64) << 24) |
        ((self.0[4] as u64) << 32) |
        ((self.0[5] as u64) << 40) |
        ((self.0[6] as u64) << 48) |
        ((self.0[7] as u64) << 56)
    }

    pub fn to_hex(&self) -> String {
        self.0.to_hex()
    }
}

impl Encodable for Fingerprint {
    #[inline]
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        for &byte in &self.0[..] {
            s.emit_u8(byte)?;
        }
        Ok(())
    }
}

impl Decodable for Fingerprint {
    #[inline]
    fn decode<D: Decoder>(d: &mut D) -> Result<Fingerprint, D::Error> {
        let mut result = Fingerprint([0u8; FINGERPRINT_LENGTH]);
        for byte in &mut result.0[..] {
            *byte = d.read_u8()?;
        }
        Ok(result)
    }
}

impl ::std::fmt::Display for Fingerprint {
    fn fmt(&self, formatter: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        for i in 0 .. self.0.len() {
            if i > 0 {
                write!(formatter, "::")?;
            }

            write!(formatter, "{}", self.0[i])?;
        }
        Ok(())
    }
}


impl stable_hasher::StableHasherResult for Fingerprint {
    fn finish(mut hasher: stable_hasher::StableHasher<Self>) -> Self {
        let mut fingerprint = Fingerprint::zero();
        fingerprint.0.copy_from_slice(hasher.finalize());
        fingerprint
    }
}
