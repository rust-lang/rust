// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Implementation of SipHash 2-4
 *
 * See: http://131002.net/siphash/
 *
 * Consider this as a main "general-purpose" hash for all hashtables: it
 * runs at good speed (competitive with spooky and city) and permits
 * strong _keyed_ hashing. Key your hashtables from a strong RNG,
 * such as `rand::Rng`.
 *
 * Although the SipHash algorithm is considered to be cryptographically
 * strong, this implementation has not been reviewed for such purposes.
 * As such, all cryptographic uses of this implementation are strongly
 * discouraged.
 */

use core::prelude::*;

use core::default::Default;

use super::{Hash, Hasher, Writer};

/// `SipState` computes a SipHash 2-4 hash over a stream of bytes.
pub struct SipState {
    k0: u64,
    k1: u64,
    length: uint, // how many bytes we've processed
    v0: u64,      // hash state
    v1: u64,
    v2: u64,
    v3: u64,
    tail: u64, // unprocessed bytes le
    ntail: uint,  // how many bytes in tail are valid
}

// sadly, these macro definitions can't appear later,
// because they're needed in the following defs;
// this design could be improved.

macro_rules! u8to64_le (
    ($buf:expr, $i:expr) =>
    ($buf[0+$i] as u64 |
     $buf[1+$i] as u64 << 8 |
     $buf[2+$i] as u64 << 16 |
     $buf[3+$i] as u64 << 24 |
     $buf[4+$i] as u64 << 32 |
     $buf[5+$i] as u64 << 40 |
     $buf[6+$i] as u64 << 48 |
     $buf[7+$i] as u64 << 56);
    ($buf:expr, $i:expr, $len:expr) =>
    ({
        let mut t = 0;
        let mut out = 0u64;
        while t < $len {
            out |= $buf[t+$i] as u64 << t*8;
            t += 1;
        }
        out
    });
)

macro_rules! rotl (
    ($x:expr, $b:expr) =>
    (($x << $b) | ($x >> (64 - $b)))
)

macro_rules! compress (
    ($v0:expr, $v1:expr, $v2:expr, $v3:expr) =>
    ({
        $v0 += $v1; $v1 = rotl!($v1, 13); $v1 ^= $v0;
        $v0 = rotl!($v0, 32);
        $v2 += $v3; $v3 = rotl!($v3, 16); $v3 ^= $v2;
        $v0 += $v3; $v3 = rotl!($v3, 21); $v3 ^= $v0;
        $v2 += $v1; $v1 = rotl!($v1, 17); $v1 ^= $v2;
        $v2 = rotl!($v2, 32);
    })
)

impl SipState {
    /// Create a `SipState` that is keyed off the provided keys.
    #[inline]
    pub fn new() -> SipState {
        SipState::new_with_keys(0, 0)
    }

    /// Create a `SipState` that is keyed off the provided keys.
    #[inline]
    pub fn new_with_keys(key0: u64, key1: u64) -> SipState {
        let mut state = SipState {
            k0: key0,
            k1: key1,
            length: 0,
            v0: 0,
            v1: 0,
            v2: 0,
            v3: 0,
            tail: 0,
            ntail: 0,
        };
        state.reset();
        state
    }

    /// Reset the state back to it's initial state.
    #[inline]
    pub fn reset(&mut self) {
        self.length = 0;
        self.v0 = self.k0 ^ 0x736f6d6570736575;
        self.v1 = self.k1 ^ 0x646f72616e646f6d;
        self.v2 = self.k0 ^ 0x6c7967656e657261;
        self.v3 = self.k1 ^ 0x7465646279746573;
        self.ntail = 0;
    }

    /// Return the computed hash.
    #[inline]
    pub fn result(&self) -> u64 {
        let mut v0 = self.v0;
        let mut v1 = self.v1;
        let mut v2 = self.v2;
        let mut v3 = self.v3;

        let b: u64 = ((self.length as u64 & 0xff) << 56) | self.tail;

        v3 ^= b;
        compress!(v0, v1, v2, v3);
        compress!(v0, v1, v2, v3);
        v0 ^= b;

        v2 ^= 0xff;
        compress!(v0, v1, v2, v3);
        compress!(v0, v1, v2, v3);
        compress!(v0, v1, v2, v3);
        compress!(v0, v1, v2, v3);

        v0 ^ v1 ^ v2 ^ v3
    }
}

impl Writer for SipState {
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        let length = msg.len();
        self.length += length;

        let mut needed = 0u;

        if self.ntail != 0 {
            needed = 8 - self.ntail;
            if length < needed {
                self.tail |= u8to64_le!(msg, 0, length) << 8*self.ntail;
                self.ntail += length;
                return
            }

            let m = self.tail | u8to64_le!(msg, 0, needed) << 8*self.ntail;

            self.v3 ^= m;
            compress!(self.v0, self.v1, self.v2, self.v3);
            compress!(self.v0, self.v1, self.v2, self.v3);
            self.v0 ^= m;

            self.ntail = 0;
        }

        // Buffered tail is now flushed, process new input.
        let len = length - needed;
        let end = len & (!0x7);
        let left = len & 0x7;

        let mut i = needed;
        while i < end {
            let mi = u8to64_le!(msg, i);

            self.v3 ^= mi;
            compress!(self.v0, self.v1, self.v2, self.v3);
            compress!(self.v0, self.v1, self.v2, self.v3);
            self.v0 ^= mi;

            i += 8;
        }

        self.tail = u8to64_le!(msg, i, left);
        self.ntail = left;
    }
}

impl Clone for SipState {
    #[inline]
    fn clone(&self) -> SipState {
        *self
    }
}

impl Default for SipState {
    #[inline]
    fn default() -> SipState {
        SipState::new()
    }
}

/// `SipHasher` computes the SipHash algorithm from a stream of bytes.
#[deriving(Clone)]
pub struct SipHasher {
    k0: u64,
    k1: u64,
}

impl SipHasher {
    /// Create a `Sip`.
    #[inline]
    pub fn new() -> SipHasher {
        SipHasher::new_with_keys(0, 0)
    }

    /// Create a `Sip` that is keyed off the provided keys.
    #[inline]
    pub fn new_with_keys(key0: u64, key1: u64) -> SipHasher {
        SipHasher {
            k0: key0,
            k1: key1,
        }
    }
}

impl Hasher<SipState> for SipHasher {
    #[inline]
    fn hash<T: Hash<SipState>>(&self, value: &T) -> u64 {
        let mut state = SipState::new_with_keys(self.k0, self.k1);
        value.hash(&mut state);
        state.result()
    }
}

impl Default for SipHasher {
    #[inline]
    fn default() -> SipHasher {
        SipHasher::new()
    }
}

/// Hash a value using the SipHash algorithm.
#[inline]
pub fn hash<T: Hash<SipState>>(value: &T) -> u64 {
    let mut state = SipState::new();
    value.hash(&mut state);
    state.result()
}

/// Hash a value with the SipHash algorithm with the provided keys.
#[inline]
pub fn hash_with_keys<T: Hash<SipState>>(k0: u64, k1: u64, value: &T) -> u64 {
    let mut state = SipState::new_with_keys(k0, k1);
    value.hash(&mut state);
    state.result()
}

#[cfg(test)]
mod tests {
    use test::Bencher;
    use std::prelude::*;
    use std::num::ToStrRadix;

    use str::Str;
    use string::String;
    use slice::{Vector, ImmutableVector};
    use vec::Vec;

    use super::super::{Hash, Writer};
    use super::{SipState, hash, hash_with_keys};

    // Hash just the bytes of the slice, without length prefix
    struct Bytes<'a>(&'a [u8]);

    impl<'a, S: Writer> Hash<S> for Bytes<'a> {
        #[allow(unused_must_use)]
        fn hash(&self, state: &mut S) {
            let Bytes(v) = *self;
            state.write(v);
        }
    }

    #[test]
    #[allow(unused_must_use)]
    fn test_siphash() {
        let vecs : [[u8, ..8], ..64] = [
            [ 0x31, 0x0e, 0x0e, 0xdd, 0x47, 0xdb, 0x6f, 0x72, ],
            [ 0xfd, 0x67, 0xdc, 0x93, 0xc5, 0x39, 0xf8, 0x74, ],
            [ 0x5a, 0x4f, 0xa9, 0xd9, 0x09, 0x80, 0x6c, 0x0d, ],
            [ 0x2d, 0x7e, 0xfb, 0xd7, 0x96, 0x66, 0x67, 0x85, ],
            [ 0xb7, 0x87, 0x71, 0x27, 0xe0, 0x94, 0x27, 0xcf, ],
            [ 0x8d, 0xa6, 0x99, 0xcd, 0x64, 0x55, 0x76, 0x18, ],
            [ 0xce, 0xe3, 0xfe, 0x58, 0x6e, 0x46, 0xc9, 0xcb, ],
            [ 0x37, 0xd1, 0x01, 0x8b, 0xf5, 0x00, 0x02, 0xab, ],
            [ 0x62, 0x24, 0x93, 0x9a, 0x79, 0xf5, 0xf5, 0x93, ],
            [ 0xb0, 0xe4, 0xa9, 0x0b, 0xdf, 0x82, 0x00, 0x9e, ],
            [ 0xf3, 0xb9, 0xdd, 0x94, 0xc5, 0xbb, 0x5d, 0x7a, ],
            [ 0xa7, 0xad, 0x6b, 0x22, 0x46, 0x2f, 0xb3, 0xf4, ],
            [ 0xfb, 0xe5, 0x0e, 0x86, 0xbc, 0x8f, 0x1e, 0x75, ],
            [ 0x90, 0x3d, 0x84, 0xc0, 0x27, 0x56, 0xea, 0x14, ],
            [ 0xee, 0xf2, 0x7a, 0x8e, 0x90, 0xca, 0x23, 0xf7, ],
            [ 0xe5, 0x45, 0xbe, 0x49, 0x61, 0xca, 0x29, 0xa1, ],
            [ 0xdb, 0x9b, 0xc2, 0x57, 0x7f, 0xcc, 0x2a, 0x3f, ],
            [ 0x94, 0x47, 0xbe, 0x2c, 0xf5, 0xe9, 0x9a, 0x69, ],
            [ 0x9c, 0xd3, 0x8d, 0x96, 0xf0, 0xb3, 0xc1, 0x4b, ],
            [ 0xbd, 0x61, 0x79, 0xa7, 0x1d, 0xc9, 0x6d, 0xbb, ],
            [ 0x98, 0xee, 0xa2, 0x1a, 0xf2, 0x5c, 0xd6, 0xbe, ],
            [ 0xc7, 0x67, 0x3b, 0x2e, 0xb0, 0xcb, 0xf2, 0xd0, ],
            [ 0x88, 0x3e, 0xa3, 0xe3, 0x95, 0x67, 0x53, 0x93, ],
            [ 0xc8, 0xce, 0x5c, 0xcd, 0x8c, 0x03, 0x0c, 0xa8, ],
            [ 0x94, 0xaf, 0x49, 0xf6, 0xc6, 0x50, 0xad, 0xb8, ],
            [ 0xea, 0xb8, 0x85, 0x8a, 0xde, 0x92, 0xe1, 0xbc, ],
            [ 0xf3, 0x15, 0xbb, 0x5b, 0xb8, 0x35, 0xd8, 0x17, ],
            [ 0xad, 0xcf, 0x6b, 0x07, 0x63, 0x61, 0x2e, 0x2f, ],
            [ 0xa5, 0xc9, 0x1d, 0xa7, 0xac, 0xaa, 0x4d, 0xde, ],
            [ 0x71, 0x65, 0x95, 0x87, 0x66, 0x50, 0xa2, 0xa6, ],
            [ 0x28, 0xef, 0x49, 0x5c, 0x53, 0xa3, 0x87, 0xad, ],
            [ 0x42, 0xc3, 0x41, 0xd8, 0xfa, 0x92, 0xd8, 0x32, ],
            [ 0xce, 0x7c, 0xf2, 0x72, 0x2f, 0x51, 0x27, 0x71, ],
            [ 0xe3, 0x78, 0x59, 0xf9, 0x46, 0x23, 0xf3, 0xa7, ],
            [ 0x38, 0x12, 0x05, 0xbb, 0x1a, 0xb0, 0xe0, 0x12, ],
            [ 0xae, 0x97, 0xa1, 0x0f, 0xd4, 0x34, 0xe0, 0x15, ],
            [ 0xb4, 0xa3, 0x15, 0x08, 0xbe, 0xff, 0x4d, 0x31, ],
            [ 0x81, 0x39, 0x62, 0x29, 0xf0, 0x90, 0x79, 0x02, ],
            [ 0x4d, 0x0c, 0xf4, 0x9e, 0xe5, 0xd4, 0xdc, 0xca, ],
            [ 0x5c, 0x73, 0x33, 0x6a, 0x76, 0xd8, 0xbf, 0x9a, ],
            [ 0xd0, 0xa7, 0x04, 0x53, 0x6b, 0xa9, 0x3e, 0x0e, ],
            [ 0x92, 0x59, 0x58, 0xfc, 0xd6, 0x42, 0x0c, 0xad, ],
            [ 0xa9, 0x15, 0xc2, 0x9b, 0xc8, 0x06, 0x73, 0x18, ],
            [ 0x95, 0x2b, 0x79, 0xf3, 0xbc, 0x0a, 0xa6, 0xd4, ],
            [ 0xf2, 0x1d, 0xf2, 0xe4, 0x1d, 0x45, 0x35, 0xf9, ],
            [ 0x87, 0x57, 0x75, 0x19, 0x04, 0x8f, 0x53, 0xa9, ],
            [ 0x10, 0xa5, 0x6c, 0xf5, 0xdf, 0xcd, 0x9a, 0xdb, ],
            [ 0xeb, 0x75, 0x09, 0x5c, 0xcd, 0x98, 0x6c, 0xd0, ],
            [ 0x51, 0xa9, 0xcb, 0x9e, 0xcb, 0xa3, 0x12, 0xe6, ],
            [ 0x96, 0xaf, 0xad, 0xfc, 0x2c, 0xe6, 0x66, 0xc7, ],
            [ 0x72, 0xfe, 0x52, 0x97, 0x5a, 0x43, 0x64, 0xee, ],
            [ 0x5a, 0x16, 0x45, 0xb2, 0x76, 0xd5, 0x92, 0xa1, ],
            [ 0xb2, 0x74, 0xcb, 0x8e, 0xbf, 0x87, 0x87, 0x0a, ],
            [ 0x6f, 0x9b, 0xb4, 0x20, 0x3d, 0xe7, 0xb3, 0x81, ],
            [ 0xea, 0xec, 0xb2, 0xa3, 0x0b, 0x22, 0xa8, 0x7f, ],
            [ 0x99, 0x24, 0xa4, 0x3c, 0xc1, 0x31, 0x57, 0x24, ],
            [ 0xbd, 0x83, 0x8d, 0x3a, 0xaf, 0xbf, 0x8d, 0xb7, ],
            [ 0x0b, 0x1a, 0x2a, 0x32, 0x65, 0xd5, 0x1a, 0xea, ],
            [ 0x13, 0x50, 0x79, 0xa3, 0x23, 0x1c, 0xe6, 0x60, ],
            [ 0x93, 0x2b, 0x28, 0x46, 0xe4, 0xd7, 0x06, 0x66, ],
            [ 0xe1, 0x91, 0x5f, 0x5c, 0xb1, 0xec, 0xa4, 0x6c, ],
            [ 0xf3, 0x25, 0x96, 0x5c, 0xa1, 0x6d, 0x62, 0x9f, ],
            [ 0x57, 0x5f, 0xf2, 0x8e, 0x60, 0x38, 0x1b, 0xe5, ],
            [ 0x72, 0x45, 0x06, 0xeb, 0x4c, 0x32, 0x8a, 0x95, ]
        ];

        let k0 = 0x_07_06_05_04_03_02_01_00_u64;
        let k1 = 0x_0f_0e_0d_0c_0b_0a_09_08_u64;
        let mut buf = Vec::new();
        let mut t = 0;
        let mut state_inc = SipState::new_with_keys(k0, k1);
        let mut state_full = SipState::new_with_keys(k0, k1);

        fn to_hex_str(r: &[u8, ..8]) -> String {
            let mut s = String::new();
            for b in r.iter() {
                s.push_str((*b as uint).to_str_radix(16u).as_slice());
            }
            s
        }

        fn result_bytes(h: u64) -> Vec<u8> {
            vec![(h >> 0) as u8,
              (h >> 8) as u8,
              (h >> 16) as u8,
              (h >> 24) as u8,
              (h >> 32) as u8,
              (h >> 40) as u8,
              (h >> 48) as u8,
              (h >> 56) as u8,
            ]
        }

        fn result_str(h: u64) -> String {
            let r = result_bytes(h);
            let mut s = String::new();
            for b in r.iter() {
                s.push_str((*b as uint).to_str_radix(16u).as_slice());
            }
            s
        }

        while t < 64 {
            debug!("siphash test {}: {}", t, buf);
            let vec = u8to64_le!(vecs[t], 0);
            let out = hash_with_keys(k0, k1, &Bytes(buf.as_slice()));
            debug!("got {:?}, expected {:?}", out, vec);
            assert_eq!(vec, out);

            state_full.reset();
            state_full.write(buf.as_slice());
            let f = result_str(state_full.result());
            let i = result_str(state_inc.result());
            let v = to_hex_str(&vecs[t]);
            debug!("{}: ({}) => inc={} full={}", t, v, i, f);

            debug!("full state {:?}", state_full);
            debug!("inc  state {:?}", state_inc);

            assert_eq!(f, i);
            assert_eq!(f, v);

            buf.push(t as u8);
            state_inc.write([t as u8]);

            t += 1;
        }
    }

    #[test] #[cfg(target_arch = "arm")]
    fn test_hash_uint() {
        let val = 0xdeadbeef_deadbeef_u64;
        assert!(hash(&(val as u64)) != hash(&(val as uint)));
        assert_eq!(hash(&(val as u32)), hash(&(val as uint)));
    }
    #[test] #[cfg(target_arch = "x86_64")]
    fn test_hash_uint() {
        let val = 0xdeadbeef_deadbeef_u64;
        assert_eq!(hash(&(val as u64)), hash(&(val as uint)));
        assert!(hash(&(val as u32)) != hash(&(val as uint)));
    }
    #[test] #[cfg(target_arch = "x86")]
    fn test_hash_uint() {
        let val = 0xdeadbeef_deadbeef_u64;
        assert!(hash(&(val as u64)) != hash(&(val as uint)));
        assert_eq!(hash(&(val as u32)), hash(&(val as uint)));
    }

    #[test]
    fn test_hash_idempotent() {
        let val64 = 0xdeadbeef_deadbeef_u64;
        assert_eq!(hash(&val64), hash(&val64));
        let val32 = 0xdeadbeef_u32;
        assert_eq!(hash(&val32), hash(&val32));
    }

    #[test]
    fn test_hash_no_bytes_dropped_64() {
        let val = 0xdeadbeef_deadbeef_u64;

        assert!(hash(&val) != hash(&zero_byte(val, 0)));
        assert!(hash(&val) != hash(&zero_byte(val, 1)));
        assert!(hash(&val) != hash(&zero_byte(val, 2)));
        assert!(hash(&val) != hash(&zero_byte(val, 3)));
        assert!(hash(&val) != hash(&zero_byte(val, 4)));
        assert!(hash(&val) != hash(&zero_byte(val, 5)));
        assert!(hash(&val) != hash(&zero_byte(val, 6)));
        assert!(hash(&val) != hash(&zero_byte(val, 7)));

        fn zero_byte(val: u64, byte: uint) -> u64 {
            assert!(byte < 8);
            val & !(0xff << (byte * 8))
        }
    }

    #[test]
    fn test_hash_no_bytes_dropped_32() {
        let val = 0xdeadbeef_u32;

        assert!(hash(&val) != hash(&zero_byte(val, 0)));
        assert!(hash(&val) != hash(&zero_byte(val, 1)));
        assert!(hash(&val) != hash(&zero_byte(val, 2)));
        assert!(hash(&val) != hash(&zero_byte(val, 3)));

        fn zero_byte(val: u32, byte: uint) -> u32 {
            assert!(byte < 4);
            val & !(0xff << (byte * 8))
        }
    }

    #[test]
    fn test_hash_no_concat_alias() {
        let s = ("aa", "bb");
        let t = ("aabb", "");
        let u = ("a", "abb");

        assert!(s != t && t != u);
        assert!(hash(&s) != hash(&t) && hash(&s) != hash(&u));

        let v = (&[1u8], &[0u8, 0], &[0u8]);
        let w = (&[1u8, 0, 0, 0], &[], &[]);

        assert!(v != w);
        assert!(hash(&v) != hash(&w));
    }

    #[bench]
    fn bench_str_under_8_bytes(b: &mut Bencher) {
        let s = "foo";
        b.iter(|| {
            assert_eq!(hash(&s), 16262950014981195938);
        })
    }

    #[bench]
    fn bench_str_of_8_bytes(b: &mut Bencher) {
        let s = "foobar78";
        b.iter(|| {
            assert_eq!(hash(&s), 4898293253460910787);
        })
    }

    #[bench]
    fn bench_str_over_8_bytes(b: &mut Bencher) {
        let s = "foobarbaz0";
        b.iter(|| {
            assert_eq!(hash(&s), 10581415515220175264);
        })
    }

    #[bench]
    fn bench_long_str(b: &mut Bencher) {
        let s = "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor \
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud \
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute \
irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla \
pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui \
officia deserunt mollit anim id est laborum.";
        b.iter(|| {
            assert_eq!(hash(&s), 17717065544121360093);
        })
    }

    #[bench]
    fn bench_u64(b: &mut Bencher) {
        let u = 16262950014981195938u64;
        b.iter(|| {
            assert_eq!(hash(&u), 5254097107239593357);
        })
    }
}
