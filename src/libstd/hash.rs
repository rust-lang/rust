// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
 * cryptographically strong _keyed_ hashing. Key your hashtables from a
 * CPRNG like rand::rng.
 */

#[allow(missing_doc)];

use container::Container;
use iterator::Iterator;
use option::{Some, None};
use rt::io::Writer;
use str::OwnedStr;
use to_bytes::IterBytes;
use uint;
use vec::ImmutableVector;

// Alias `SipState` to `State`.
pub use State = hash::SipState;

/**
 * Types that can meaningfully be hashed should implement this.
 *
 * Note that this trait is likely to change somewhat as it is
 * closely related to `to_bytes::IterBytes` and in almost all
 * cases presently the two are (and must be) used together.
 *
 * In general, most types only need to implement `IterBytes`,
 * and the implementation of `Hash` below will take care of
 * the rest. This is the recommended approach, since constructing
 * good keyed hash functions is quite difficult.
 */
pub trait Hash {
    /**
     * Compute a "keyed" hash of the value implementing the trait,
     * taking `k0` and `k1` as "keying" parameters that randomize or
     * otherwise perturb the hash function in such a way that a
     * hash table built using such "keyed hash functions" cannot
     * be made to perform linearly by an attacker controlling the
     * hashtable's contents.
     *
     * In practical terms, we implement this using the SipHash 2-4
     * function and require most types to only implement the
     * IterBytes trait, that feeds SipHash.
     */
    fn hash_keyed(&self, k0: u64, k1: u64) -> u64;
}

// When we have default methods, won't need this.
pub trait HashUtil {
    fn hash(&self) -> u64;
}

impl<A:Hash> HashUtil for A {
    #[inline]
    fn hash(&self) -> u64 { self.hash_keyed(0,0) }
}

/// Streaming hash-functions should implement this.
pub trait Streaming {
    fn input(&mut self, &[u8]);
    // These can be refactored some when we have default methods.
    fn result_bytes(&mut self) -> ~[u8];
    fn result_str(&mut self) -> ~str;
    fn result_u64(&mut self) -> u64;
    fn reset(&mut self);
}

impl<A:IterBytes> Hash for A {
    #[inline]
    fn hash_keyed(&self, k0: u64, k1: u64) -> u64 {
        let mut s = State::new(k0, k1);
        do self.iter_bytes(true) |bytes| {
            s.input(bytes);
            true
        };
        s.result_u64()
    }
}

fn hash_keyed_2<A: IterBytes,
                B: IterBytes>(a: &A, b: &B, k0: u64, k1: u64) -> u64 {
    let mut s = State::new(k0, k1);
    do a.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do b.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    s.result_u64()
}

fn hash_keyed_3<A: IterBytes,
                B: IterBytes,
                C: IterBytes>(a: &A, b: &B, c: &C, k0: u64, k1: u64) -> u64 {
    let mut s = State::new(k0, k1);
    do a.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do b.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do c.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    s.result_u64()
}

fn hash_keyed_4<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes>(
                a: &A,
                b: &B,
                c: &C,
                d: &D,
                k0: u64,
                k1: u64)
                -> u64 {
    let mut s = State::new(k0, k1);
    do a.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do b.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do c.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do d.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    s.result_u64()
}

fn hash_keyed_5<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes,
                E: IterBytes>(
                a: &A,
                b: &B,
                c: &C,
                d: &D,
                e: &E,
                k0: u64,
                k1: u64)
                -> u64 {
    let mut s = State::new(k0, k1);
    do a.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do b.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do c.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do d.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    do e.iter_bytes(true) |bytes| {
        s.input(bytes);
        true
    };
    s.result_u64()
}

#[inline]
pub fn default_state() -> State {
    State::new(0, 0)
}

struct SipState {
    k0: u64,
    k1: u64,
    length: uint, // how many bytes we've processed
    v0: u64,      // hash state
    v1: u64,
    v2: u64,
    v3: u64,
    tail: [u8, ..8], // unprocessed bytes
    ntail: uint,  // how many bytes in tail are valid
}

impl SipState {
    #[inline]
    fn new(key0: u64, key1: u64) -> SipState {
        let mut state = SipState {
            k0: key0,
            k1: key1,
            length: 0,
            v0: 0,
            v1: 0,
            v2: 0,
            v3: 0,
            tail: [ 0, 0, 0, 0, 0, 0, 0, 0 ],
            ntail: 0,
        };
        state.reset();
        state
    }
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
     $buf[7+$i] as u64 << 56)
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


impl Writer for SipState {
    // Methods for io::writer
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        let length = msg.len();
        self.length += length;

        let mut needed = 0u;

        if self.ntail != 0 {
            needed = 8 - self.ntail;

            if length < needed {
                let mut t = 0;
                while t < length {
                    self.tail[self.ntail+t] = msg[t];
                    t += 1;
                }
                self.ntail += length;
                return;
            }

            let mut t = 0;
            while t < needed {
                self.tail[self.ntail+t] = msg[t];
                t += 1;
            }

            let m = u8to64_le!(self.tail, 0);

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

        let mut t = 0u;
        while t < left {
            self.tail[t] = msg[i+t];
            t += 1
        }
        self.ntail = left;
    }

    fn flush(&mut self) {
        // No-op
    }
}

impl Streaming for SipState {
    #[inline]
    fn input(&mut self, buf: &[u8]) {
        self.write(buf);
    }

    #[inline]
    fn result_u64(&mut self) -> u64 {
        let mut v0 = self.v0;
        let mut v1 = self.v1;
        let mut v2 = self.v2;
        let mut v3 = self.v3;

        let mut b : u64 = (self.length as u64 & 0xff) << 56;

        if self.ntail > 0 { b |= self.tail[0] as u64 <<  0; }
        if self.ntail > 1 { b |= self.tail[1] as u64 <<  8; }
        if self.ntail > 2 { b |= self.tail[2] as u64 << 16; }
        if self.ntail > 3 { b |= self.tail[3] as u64 << 24; }
        if self.ntail > 4 { b |= self.tail[4] as u64 << 32; }
        if self.ntail > 5 { b |= self.tail[5] as u64 << 40; }
        if self.ntail > 6 { b |= self.tail[6] as u64 << 48; }

        v3 ^= b;
        compress!(v0, v1, v2, v3);
        compress!(v0, v1, v2, v3);
        v0 ^= b;

        v2 ^= 0xff;
        compress!(v0, v1, v2, v3);
        compress!(v0, v1, v2, v3);
        compress!(v0, v1, v2, v3);
        compress!(v0, v1, v2, v3);

        return (v0 ^ v1 ^ v2 ^ v3);
    }

    fn result_bytes(&mut self) -> ~[u8] {
        let h = self.result_u64();
        ~[(h >> 0) as u8,
          (h >> 8) as u8,
          (h >> 16) as u8,
          (h >> 24) as u8,
          (h >> 32) as u8,
          (h >> 40) as u8,
          (h >> 48) as u8,
          (h >> 56) as u8,
        ]
    }

    fn result_str(&mut self) -> ~str {
        let r = self.result_bytes();
        let mut s = ~"";
        for b in r.iter() {
            s.push_str(uint::to_str_radix(*b as uint, 16u));
        }
        s
    }

    #[inline]
    fn reset(&mut self) {
        self.length = 0;
        self.v0 = self.k0 ^ 0x736f6d6570736575;
        self.v1 = self.k1 ^ 0x646f72616e646f6d;
        self.v2 = self.k0 ^ 0x6c7967656e657261;
        self.v3 = self.k1 ^ 0x7465646279746573;
        self.ntail = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    use uint;

    #[test]
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
        let mut buf : ~[u8] = ~[];
        let mut t = 0;
        let mut stream_inc = SipState::new(k0, k1);
        let mut stream_full = SipState::new(k0, k1);

        fn to_hex_str(r: &[u8, ..8]) -> ~str {
            let mut s = ~"";
            for b in r.iter() {
                s.push_str(uint::to_str_radix(*b as uint, 16u));
            }
            s
        }

        while t < 64 {
            debug!("siphash test %?", t);
            let vec = u8to64_le!(vecs[t], 0);
            let out = buf.hash_keyed(k0, k1);
            debug!("got %?, expected %?", out, vec);
            assert_eq!(vec, out);

            stream_full.reset();
            stream_full.input(buf);
            let f = stream_full.result_str();
            let i = stream_inc.result_str();
            let v = to_hex_str(&vecs[t]);
            debug!("%d: (%s) => inc=%s full=%s", t, v, i, f);

            assert!(f == i && f == v);

            buf.push(t as u8);
            stream_inc.input([t as u8]);

            t += 1;
        }
    }

    #[test] #[cfg(target_arch = "arm")]
    fn test_hash_uint() {
        let val = 0xdeadbeef_deadbeef_u64;
        assert!((val as u64).hash() != (val as uint).hash());
        assert_eq!((val as u32).hash(), (val as uint).hash());
    }
    #[test] #[cfg(target_arch = "x86_64")]
    fn test_hash_uint() {
        let val = 0xdeadbeef_deadbeef_u64;
        assert_eq!((val as u64).hash(), (val as uint).hash());
        assert!((val as u32).hash() != (val as uint).hash());
    }
    #[test] #[cfg(target_arch = "x86")]
    fn test_hash_uint() {
        let val = 0xdeadbeef_deadbeef_u64;
        assert!((val as u64).hash() != (val as uint).hash());
        assert_eq!((val as u32).hash(), (val as uint).hash());
    }

    #[test]
    fn test_hash_idempotent() {
        let val64 = 0xdeadbeef_deadbeef_u64;
        val64.hash() == val64.hash();
        let val32 = 0xdeadbeef_u32;
        val32.hash() == val32.hash();
    }

    #[test]
    fn test_hash_no_bytes_dropped_64() {
        let val = 0xdeadbeef_deadbeef_u64;

        assert!(val.hash() != zero_byte(val, 0).hash());
        assert!(val.hash() != zero_byte(val, 1).hash());
        assert!(val.hash() != zero_byte(val, 2).hash());
        assert!(val.hash() != zero_byte(val, 3).hash());
        assert!(val.hash() != zero_byte(val, 4).hash());
        assert!(val.hash() != zero_byte(val, 5).hash());
        assert!(val.hash() != zero_byte(val, 6).hash());
        assert!(val.hash() != zero_byte(val, 7).hash());

        fn zero_byte(val: u64, byte: uint) -> u64 {
            assert!(byte < 8);
            val & !(0xff << (byte * 8))
        }
    }

    #[test]
    fn test_hash_no_bytes_dropped_32() {
        let val = 0xdeadbeef_u32;

        assert!(val.hash() != zero_byte(val, 0).hash());
        assert!(val.hash() != zero_byte(val, 1).hash());
        assert!(val.hash() != zero_byte(val, 2).hash());
        assert!(val.hash() != zero_byte(val, 3).hash());

        fn zero_byte(val: u32, byte: uint) -> u32 {
            assert!(byte < 4);
            val & !(0xff << (byte * 8))
        }
    }

    #[test]
    fn test_float_hashes_differ() {
        assert!(0.0.hash() != 1.0.hash());
        assert!(1.0.hash() != (-1.0).hash());
    }

    #[test]
    fn test_float_hashes_of_zero() {
        assert_eq!(0.0.hash(), (-0.0).hash());
    }
}
