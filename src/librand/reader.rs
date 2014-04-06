// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A wrapper around any Reader to treat it as an RNG.

use Rng;

/// An RNG that reads random bytes straight from a `Reader`. This will
/// work best with an infinite reader, but this is not required.
///
/// It will fail if it there is insufficient data to fulfill a request.
///
/// # Example
///
/// ```rust
/// use rand::{reader, Rng};
/// use std::io::MemReader;
///
/// let mut rng = reader::ReaderRng::new(MemReader::new(vec!(1,2,3,4,5,6,7,8)));
/// println!("{:x}", rng.gen::<uint>());
/// ```
pub struct ReaderRng<R> {
    reader: R
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
        // This is designed for speed: reading a LE integer on a LE
        // platform just involves blitting the bytes into the memory
        // of the u32, similarly for BE on BE; avoiding byteswapping.
        if cfg!(target_endian="little") {
            self.reader.read_le_u32().unwrap()
        } else {
            self.reader.read_be_u32().unwrap()
        }
    }
    fn next_u64(&mut self) -> u64 {
        // see above for explanation.
        if cfg!(target_endian="little") {
            self.reader.read_le_u64().unwrap()
        } else {
            self.reader.read_be_u64().unwrap()
        }
    }
    fn fill_bytes(&mut self, v: &mut [u8]) {
        if v.len() == 0 { return }
        match self.reader.fill(v) {
            Ok(()) => {}
            Err(e) => fail!("ReaderRng.fill_bytes error: {}", e)
        }
    }
}

#[cfg(test)]
#[allow(deprecated_owned_vector)]
mod test {
    use super::ReaderRng;
    use std::io::MemReader;
    use std::cast;
    use Rng;

    #[test]
    fn test_reader_rng_u64() {
        // transmute from the target to avoid endianness concerns.
        let v = ~[1u64, 2u64, 3u64];
        let bytes: ~[u8] = unsafe {cast::transmute(v)};
        let mut rng = ReaderRng::new(MemReader::new(bytes.move_iter().collect()));

        assert_eq!(rng.next_u64(), 1);
        assert_eq!(rng.next_u64(), 2);
        assert_eq!(rng.next_u64(), 3);
    }
    #[test]
    fn test_reader_rng_u32() {
        // transmute from the target to avoid endianness concerns.
        let v = ~[1u32, 2u32, 3u32];
        let bytes: ~[u8] = unsafe {cast::transmute(v)};
        let mut rng = ReaderRng::new(MemReader::new(bytes.move_iter().collect()));

        assert_eq!(rng.next_u32(), 1);
        assert_eq!(rng.next_u32(), 2);
        assert_eq!(rng.next_u32(), 3);
    }
    #[test]
    fn test_reader_rng_fill_bytes() {
        let v = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut w = [0u8, .. 8];

        let mut rng = ReaderRng::new(MemReader::new(Vec::from_slice(v)));
        rng.fill_bytes(w);

        assert!(v == w);
    }

    #[test]
    #[should_fail]
    fn test_reader_rng_insufficient_bytes() {
        let mut rng = ReaderRng::new(MemReader::new(vec!()));
        let mut v = [0u8, .. 3];
        rng.fill_bytes(v);
    }
}
