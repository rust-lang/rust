// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use rustc_data_structures::blake2b::Blake2bHasher;
use rustc::ty::util::ArchIndependentHasher;
use ich::Fingerprint;

#[derive(Debug)]
pub struct IchHasher {
    state: ArchIndependentHasher<Blake2bHasher>,
    bytes_hashed: u64,
}

impl IchHasher {
    pub fn new() -> IchHasher {
        let hash_size = mem::size_of::<Fingerprint>();
        IchHasher {
            state: ArchIndependentHasher::new(Blake2bHasher::new(hash_size, &[])),
            bytes_hashed: 0
        }
    }

    pub fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }

    pub fn finish(self) -> Fingerprint {
        let mut fingerprint = Fingerprint::zero();
        fingerprint.0.copy_from_slice(self.state.into_inner().finalize());
        fingerprint
    }
}

impl ::std::hash::Hasher for IchHasher {
    fn finish(&self) -> u64 {
        bug!("Use other finish() implementation to get the full 128-bit hash.");
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
        self.bytes_hashed += bytes.len() as u64;
    }
}
