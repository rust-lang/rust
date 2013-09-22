// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rt::io::Reader;
use rt::io::ReaderByteConversions;

use rand::Rng;

/// An RNG that reads random bytes straight from a `Reader`. This will
/// work best with an infinite reader, but this is not required. The
/// semantics of reading past the end of the reader are the same as
/// those of the `read` method of the inner `Reader`.
pub struct ReaderRng<R> {
    priv reader: R
}

impl<R: Reader> ReaderRng<R> {
    /// Create a new `ReaderRng` from a `Reader`.
    pub fn new(r: R) -> ReaderRng<R> {
        ReaderRng {
            reader: r
        }
    }
}

impl<R: Reader> Rng for ReaderRng<R> {
    fn next_u32(&mut self) -> u32 {
        // XXX which is better: consistency between big/little-endian
        // platforms, or speed.
        if cfg!(target_endian="little") {
            self.reader.read_le_u32_()
        } else {
            self.reader.read_be_u32_()
        }
    }
    fn next_u64(&mut self) -> u64 {
        if cfg!(target_endian="little") {
            self.reader.read_le_u64_()
        } else {
            self.reader.read_be_u64_()
        }
    }
    fn fill_bytes(&mut self, v: &mut [u8]) {
        // XXX: check that we filled `v``
        let _n = self.reader.read(v);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rt::io::mem::MemReader;
    use cast;

    #[test]
    fn test_reader_rng_u64() {
        // transmute from the target to avoid endianness concerns.
        let v = ~[1u64, 2u64, 3u64];
        let bytes: ~[u8] = unsafe {cast::transmute(v)};
        let mut rng = ReaderRng::new(MemReader::new(bytes));

        assert_eq!(rng.next_u64(), 1);
        assert_eq!(rng.next_u64(), 2);
        assert_eq!(rng.next_u64(), 3);
    }
    #[test]
    fn test_reader_rng_u32() {
        // transmute from the target to avoid endianness concerns.
        let v = ~[1u32, 2u32, 3u32];
        let bytes: ~[u8] = unsafe {cast::transmute(v)};
        let mut rng = ReaderRng::new(MemReader::new(bytes));

        assert_eq!(rng.next_u32(), 1);
        assert_eq!(rng.next_u32(), 2);
        assert_eq!(rng.next_u32(), 3);
    }
    #[test]
    fn test_reader_rng_fill_bytes() {
        let v = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut w = [0u8, .. 8];

        let mut rng = ReaderRng::new(MemReader::new(v.to_owned()));
        rng.fill_bytes(w);

        assert_eq!(v, w);
    }
}
