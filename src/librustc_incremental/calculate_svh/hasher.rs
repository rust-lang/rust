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
use std::collections::hash_map::DefaultHasher;

#[derive(Debug)]
pub struct IchHasher {
    // FIXME: this should use SHA1, not DefaultHasher. DefaultHasher is not
    // built to avoid collisions.
    state: DefaultHasher,
    bytes_hashed: u64,
}

impl IchHasher {
    pub fn new() -> IchHasher {
        IchHasher {
            state: DefaultHasher::new(),
            bytes_hashed: 0
        }
    }

    pub fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }
}

impl Hasher for IchHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state.finish()
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
        self.bytes_hashed += bytes.len() as u64;
    }
}
