// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generic hashing support.

pub use core_collections::hash::{Hash, Hasher, Writer, hash, sip};

use default::Default;
use rand::Rng;
use rand;

/// `RandomSipHasher` computes the SipHash algorithm from a stream of bytes
/// initialized with random keys.
#[deriving(Clone)]
pub struct RandomSipHasher {
    hasher: sip::SipHasher,
}

impl RandomSipHasher {
    /// Construct a new `RandomSipHasher` that is initialized with random keys.
    #[inline]
    pub fn new() -> RandomSipHasher {
        let mut r = rand::task_rng();
        let r0 = r.gen();
        let r1 = r.gen();
        RandomSipHasher {
            hasher: sip::SipHasher::new_with_keys(r0, r1),
        }
    }
}

impl Hasher<sip::SipState> for RandomSipHasher {
    #[inline]
    fn hash<T: Hash<sip::SipState>>(&self, value: &T) -> u64 {
        self.hasher.hash(value)
    }
}

impl Default for RandomSipHasher {
    #[inline]
    fn default() -> RandomSipHasher {
        RandomSipHasher::new()
    }
}
