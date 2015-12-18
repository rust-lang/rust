// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A wrapper around any Read to treat it as an RNG.

#![allow(dead_code)]

use io::prelude::*;
use rand::Rng;

/// An RNG that reads random bytes straight from a `Read`. This will
/// work best with an infinite reader, but this is not required.
///
/// # Panics
///
/// It will panic if it there is insufficient data to fulfill a request.
pub struct ReaderRng<R> {
    reader: R
}

impl<R: Read> ReaderRng<R> {
    /// Create a new `ReaderRng` from a `Read`.
    pub fn new(r: R) -> ReaderRng<R> {
        ReaderRng {
            reader: r
        }
    }
}

impl<R: Read> Rng for ReaderRng<R> {
    fn next_u32(&mut self) -> u32 {
        // This is designed for speed: reading a LE integer on a LE
        // platform just involves blitting the bytes into the memory
        // of the u32, similarly for BE on BE; avoiding byteswapping.
        let mut bytes = [0; 4];
        self.fill_bytes(&mut bytes);
        unsafe { *(bytes.as_ptr() as *const u32) }
    }
    fn next_u64(&mut self) -> u64 {
        // see above for explanation.
        let mut bytes = [0; 8];
        self.fill_bytes(&mut bytes);
        unsafe { *(bytes.as_ptr() as *const u64) }
    }
    fn fill_bytes(&mut self, mut v: &mut [u8]) {
        while !v.is_empty() {
            let t = v;
            match self.reader.read(t) {
                Ok(0) => panic!("ReaderRng.fill_bytes: EOF reached"),
                Ok(n) => v = t.split_at_mut(n).1,
                Err(e) => panic!("ReaderRng.fill_bytes: {}", e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReaderRng;
    use rand::Rng;

    #[test]
    fn test_reader_rng_u64() {
        // transmute from the target to avoid endianness concerns.
        let v = &[0, 0, 0, 0, 0, 0, 0, 1,
                  0, 0, 0, 0, 0, 0, 0, 2,
                  0, 0, 0, 0, 0, 0, 0, 3][..];
        let mut rng = ReaderRng::new(v);

        assert_eq!(rng.next_u64(), 1u64.to_be());
        assert_eq!(rng.next_u64(), 2u64.to_be());
        assert_eq!(rng.next_u64(), 3u64.to_be());
    }
    #[test]
    fn test_reader_rng_u32() {
        let v = &[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3][..];
        let mut rng = ReaderRng::new(v);

        assert_eq!(rng.next_u32(), 1u32.to_be());
        assert_eq!(rng.next_u32(), 2u32.to_be());
        assert_eq!(rng.next_u32(), 3u32.to_be());
    }
    #[test]
    fn test_reader_rng_fill_bytes() {
        let v = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut w = [0; 8];

        let mut rng = ReaderRng::new(&v[..]);
        rng.fill_bytes(&mut w);

        assert!(v == w);
    }

    #[test]
    #[should_panic]
    fn test_reader_rng_insufficient_bytes() {
        let mut rng = ReaderRng::new(&[][..]);
        let mut v = [0; 3];
        rng.fill_bytes(&mut v);
    }
}
