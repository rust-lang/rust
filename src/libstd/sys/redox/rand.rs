// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fs::File;
use io;
use rand::Rng;
use rand::reader::ReaderRng;

pub struct OsRng {
    inner: ReaderRng<File>,
}

impl OsRng {
    /// Create a new `OsRng`.
    pub fn new() -> io::Result<OsRng> {
        let reader = File::open("rand:")?;
        let reader_rng = ReaderRng::new(reader);

        Ok(OsRng { inner: reader_rng })
    }
}

impl Rng for OsRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }
    fn fill_bytes(&mut self, v: &mut [u8]) {
        self.inner.fill_bytes(v)
    }
}
