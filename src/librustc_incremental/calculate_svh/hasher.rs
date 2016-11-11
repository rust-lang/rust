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
use std::hash::Hasher;
use rustc_data_structures::blake2b::Blake2bHasher;
use rustc::ty::util::ArchIndependentHasher;
use ich::Fingerprint;
use rustc_serialize::leb128::write_unsigned_leb128;

#[derive(Debug)]
pub struct IchHasher {
    state: ArchIndependentHasher<Blake2bHasher>,
    leb128_helper: Vec<u8>,
    bytes_hashed: u64,
}

impl IchHasher {
    pub fn new() -> IchHasher {
        let hash_size = mem::size_of::<Fingerprint>();
        IchHasher {
            state: ArchIndependentHasher::new(Blake2bHasher::new(hash_size, &[])),
            leb128_helper: vec![],
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

    #[inline]
    fn write_uleb128(&mut self, value: u64) {
        let len = write_unsigned_leb128(&mut self.leb128_helper, 0, value);
        self.state.write(&self.leb128_helper[0..len]);
        self.bytes_hashed += len as u64;
    }
}

// For the non-u8 integer cases we leb128 encode them first. Because small
// integers dominate, this significantly and cheaply reduces the number of
// bytes hashed, which is good because blake2b is expensive.
impl Hasher for IchHasher {
    fn finish(&self) -> u64 {
        bug!("Use other finish() implementation to get the full 128-bit hash.");
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
        self.bytes_hashed += bytes.len() as u64;
    }

    // There is no need to leb128-encode u8 values.

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.write_uleb128(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_uleb128(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.write_uleb128(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_uleb128(i as u64);
    }
}
