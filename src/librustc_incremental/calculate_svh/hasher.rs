// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hash::Hasher;
use std::mem;
use rustc_data_structures::blake2b;
use ich::Fingerprint;

#[derive(Debug)]
pub struct IchHasher {
    state: blake2b::Blake2bCtx,
    bytes_hashed: u64,
}

impl IchHasher {
    pub fn new() -> IchHasher {
        IchHasher {
            state: blake2b::blake2b_new(mem::size_of::<Fingerprint>(), &[]),
            bytes_hashed: 0
        }
    }

    pub fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }

    pub fn finish(self) -> Fingerprint {
        let mut fingerprint = Fingerprint::zero();
        blake2b::blake2b_final(self.state, &mut fingerprint.0);
        fingerprint
    }
}

impl Hasher for IchHasher {
    fn finish(&self) -> u64 {
        bug!("Use other finish() implementation to get the full 128-bit hash.");
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        blake2b::blake2b_update(&mut self.state, bytes);
        self.bytes_hashed += bytes.len() as u64;
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.write(&unsafe { mem::transmute::<_, [u8; 2]>(i.to_le()) })
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write(&unsafe { mem::transmute::<_, [u8; 4]>(i.to_le()) })
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.write(&unsafe { mem::transmute::<_, [u8; 8]>(i.to_le()) })
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        // always hash as u64, so we don't depend on the size of `usize`
        self.write_u64(i as u64);
    }
}
