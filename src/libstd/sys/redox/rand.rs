// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use libc;
use rand::Rng;

pub struct OsRng;

impl OsRng {
    /// Create a new `OsRng`.
    pub fn new() -> io::Result<OsRng> {
        Ok(OsRng)
    }
}

impl Rng for OsRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn next_u64(&mut self) -> u64 {
        unsafe { libc::random() }
    }
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let mut rand: u64 = self.next_u64();
            for b in chunk.iter_mut() {
                *b = rand as u8;
                rand = rand >> 8;
            }
        }
    }
}
