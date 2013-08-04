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
 * An implementation of the SHA-1 cryptographic hash.
 *
 * First create a `sha1` object using the `sha1` constructor, then
 * feed it input using the `input` or `input_str` methods, which may be
 * called any number of times.
 *
 * After the entire input has been fed to the hash read the result using
 * the `result` or `result_str` methods.
 *
 * The `sha1` object may be reused to create multiple hashes by calling
 * the `reset` method.
 */


use cryptoutil::{write_u32_be, read_u32v_be, shift_add_check_overflow, FixedBuffer, FixedBuffer64,
    StandardPadding};
use digest::Digest;

/*
 * A SHA-1 implementation derived from Paul E. Jones's reference
 * implementation, which is written for clarity, not speed. At some
 * point this will want to be rewritten.
 */

// Some unexported constants
static DIGEST_BUF_LEN: uint = 5u;
static WORK_BUF_LEN: uint = 80u;
static K0: u32 = 0x5A827999u32;
static K1: u32 = 0x6ED9EBA1u32;
static K2: u32 = 0x8F1BBCDCu32;
static K3: u32 = 0xCA62C1D6u32;

/// Structure representing the state of a Sha1 computation
pub struct Sha1 {
    priv h: [u32, ..DIGEST_BUF_LEN],
    priv length_bits: u64,
    priv buffer: FixedBuffer64,
    priv computed: bool,
}

fn add_input(st: &mut Sha1, msg: &[u8]) {
    assert!((!st.computed));
    // Assumes that msg.len() can be converted to u64 without overflow
    st.length_bits = shift_add_check_overflow(st.length_bits, msg.len() as u64, 3);
    st.buffer.input(msg, |d: &[u8]| { process_msg_block(d, &mut st.h); });
}

fn process_msg_block(data: &[u8], h: &mut [u32, ..DIGEST_BUF_LEN]) {
    let mut t: int; // Loop counter

    let mut w = [0u32, ..WORK_BUF_LEN];

    // Initialize the first 16 words of the vector w
    read_u32v_be(w.mut_slice(0, 16), data);

    // Initialize the rest of vector w
    t = 16;
    while t < 80 {
        let val = w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16];
        w[t] = circular_shift(1, val);
        t += 1;
    }
    let mut a = h[0];
    let mut b = h[1];
    let mut c = h[2];
    let mut d = h[3];
    let mut e = h[4];
    let mut temp: u32;
    t = 0;
    while t < 20 {
        temp = circular_shift(5, a) + (b & c | !b & d) + e + w[t] + K0;
        e = d;
        d = c;
        c = circular_shift(30, b);
        b = a;
        a = temp;
        t += 1;
    }
    while t < 40 {
        temp = circular_shift(5, a) + (b ^ c ^ d) + e + w[t] + K1;
        e = d;
        d = c;
        c = circular_shift(30, b);
        b = a;
        a = temp;
        t += 1;
    }
    while t < 60 {
        temp =
            circular_shift(5, a) + (b & c | b & d | c & d) + e + w[t] +
                K2;
        e = d;
        d = c;
        c = circular_shift(30, b);
        b = a;
        a = temp;
        t += 1;
    }
    while t < 80 {
        temp = circular_shift(5, a) + (b ^ c ^ d) + e + w[t] + K3;
        e = d;
        d = c;
        c = circular_shift(30, b);
        b = a;
        a = temp;
        t += 1;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
}

fn circular_shift(bits: u32, word: u32) -> u32 {
    return word << bits | word >> 32u32 - bits;
}

fn mk_result(st: &mut Sha1, rs: &mut [u8]) {
    if !st.computed {
        st.buffer.standard_padding(8, |d: &[u8]| { process_msg_block(d, &mut st.h) });
        write_u32_be(st.buffer.next(4), (st.length_bits >> 32) as u32 );
        write_u32_be(st.buffer.next(4), st.length_bits as u32);
        process_msg_block(st.buffer.full_buffer(), &mut st.h);

        st.computed = true;
    }

    write_u32_be(rs.mut_slice(0, 4), st.h[0]);
    write_u32_be(rs.mut_slice(4, 8), st.h[1]);
    write_u32_be(rs.mut_slice(8, 12), st.h[2]);
    write_u32_be(rs.mut_slice(12, 16), st.h[3]);
    write_u32_be(rs.mut_slice(16, 20), st.h[4]);
}

impl Sha1 {
    /// Construct a `sha` object
    pub fn new() -> Sha1 {
        let mut st = Sha1 {
            h: [0u32, ..DIGEST_BUF_LEN],
            length_bits: 0u64,
            buffer: FixedBuffer64::new(),
            computed: false,
        };
        st.reset();
        return st;
    }
}

impl Digest for Sha1 {
    pub fn reset(&mut self) {
        self.length_bits = 0;
        self.h[0] = 0x67452301u32;
        self.h[1] = 0xEFCDAB89u32;
        self.h[2] = 0x98BADCFEu32;
        self.h[3] = 0x10325476u32;
        self.h[4] = 0xC3D2E1F0u32;
        self.buffer.reset();
        self.computed = false;
    }
    pub fn input(&mut self, msg: &[u8]) { add_input(self, msg); }
    pub fn result(&mut self, out: &mut [u8]) { return mk_result(self, out); }
    pub fn output_bits(&self) -> uint { 160 }
}

#[cfg(test)]
mod tests {
    use cryptoutil::test::test_digest_1million_random;
    use digest::Digest;
    use sha1::Sha1;

    #[deriving(Clone)]
    struct Test {
        input: ~str,
        output: ~[u8],
        output_str: ~str,
    }

    #[test]
    fn test() {
        // Test messages from FIPS 180-1

        let fips_180_1_tests = ~[
            Test {
                input: ~"abc",
                output: ~[
                    0xA9u8, 0x99u8, 0x3Eu8, 0x36u8,
                    0x47u8, 0x06u8, 0x81u8, 0x6Au8,
                    0xBAu8, 0x3Eu8, 0x25u8, 0x71u8,
                    0x78u8, 0x50u8, 0xC2u8, 0x6Cu8,
                    0x9Cu8, 0xD0u8, 0xD8u8, 0x9Du8,
                ],
                output_str: ~"a9993e364706816aba3e25717850c26c9cd0d89d"
            },
            Test {
                input:
                     ~"abcdbcdecdefdefgefghfghighij" +
                     "hijkijkljklmklmnlmnomnopnopq",
                output: ~[
                    0x84u8, 0x98u8, 0x3Eu8, 0x44u8,
                    0x1Cu8, 0x3Bu8, 0xD2u8, 0x6Eu8,
                    0xBAu8, 0xAEu8, 0x4Au8, 0xA1u8,
                    0xF9u8, 0x51u8, 0x29u8, 0xE5u8,
                    0xE5u8, 0x46u8, 0x70u8, 0xF1u8,
                ],
                output_str: ~"84983e441c3bd26ebaae4aa1f95129e5e54670f1"
            },
        ];
        // Examples from wikipedia

        let wikipedia_tests = ~[
            Test {
                input: ~"The quick brown fox jumps over the lazy dog",
                output: ~[
                    0x2fu8, 0xd4u8, 0xe1u8, 0xc6u8,
                    0x7au8, 0x2du8, 0x28u8, 0xfcu8,
                    0xedu8, 0x84u8, 0x9eu8, 0xe1u8,
                    0xbbu8, 0x76u8, 0xe7u8, 0x39u8,
                    0x1bu8, 0x93u8, 0xebu8, 0x12u8,
                ],
                output_str: ~"2fd4e1c67a2d28fced849ee1bb76e7391b93eb12",
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy cog",
                output: ~[
                    0xdeu8, 0x9fu8, 0x2cu8, 0x7fu8,
                    0xd2u8, 0x5eu8, 0x1bu8, 0x3au8,
                    0xfau8, 0xd3u8, 0xe8u8, 0x5au8,
                    0x0bu8, 0xd1u8, 0x7du8, 0x9bu8,
                    0x10u8, 0x0du8, 0xb4u8, 0xb3u8,
                ],
                output_str: ~"de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3",
            },
        ];
        let tests = fips_180_1_tests + wikipedia_tests;

        // Test that it works when accepting the message all at once

        let mut out = [0u8, ..20];

        let mut sh = ~Sha1::new();
        for t in tests.iter() {
            (*sh).input_str(t.input);
            sh.result(out);
            assert!(t.output.as_slice() == out);

            let out_str = (*sh).result_str();
            assert_eq!(out_str.len(), 40);
            assert!(out_str == t.output_str);

            sh.reset();
        }


        // Test that it works when accepting the message in pieces
        for t in tests.iter() {
            let len = t.input.len();
            let mut left = len;
            while left > 0u {
                let take = (left + 1u) / 2u;
                (*sh).input_str(t.input.slice(len - left, take + len - left));
                left = left - take;
            }
            sh.result(out);
            assert!(t.output.as_slice() == out);

            let out_str = (*sh).result_str();
            assert_eq!(out_str.len(), 40);
            assert!(out_str == t.output_str);

            sh.reset();
        }
    }

    #[test]
    fn test_1million_random_sha1() {
        let mut sh = Sha1::new();
        test_digest_1million_random(
            &mut sh,
            64,
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    }
}

#[cfg(test)]
mod bench {

    use sha1::Sha1;
    use test::BenchHarness;

    #[bench]
    pub fn sha1_10(bh: & mut BenchHarness) {
        let mut sh = Sha1::new();
        let bytes = [1u8, ..10];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn sha1_1k(bh: & mut BenchHarness) {
        let mut sh = Sha1::new();
        let bytes = [1u8, ..1024];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn sha1_64k(bh: & mut BenchHarness) {
        let mut sh = Sha1::new();
        let bytes = [1u8, ..65536];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

}
