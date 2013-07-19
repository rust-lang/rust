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


use digest::Digest;

/*
 * A SHA-1 implementation derived from Paul E. Jones's reference
 * implementation, which is written for clarity, not speed. At some
 * point this will want to be rewritten.
 */

// Some unexported constants
static DIGEST_BUF_LEN: uint = 5u;
static MSG_BLOCK_LEN: uint = 64u;
static WORK_BUF_LEN: uint = 80u;
static K0: u32 = 0x5A827999u32;
static K1: u32 = 0x6ED9EBA1u32;
static K2: u32 = 0x8F1BBCDCu32;
static K3: u32 = 0xCA62C1D6u32;

/// Structure representing the state of a Sha1 computation
pub struct Sha1 {
    priv h: [u32, ..DIGEST_BUF_LEN],
    priv len_low: u32,
    priv len_high: u32,
    priv msg_block: [u8, ..MSG_BLOCK_LEN],
    priv msg_block_idx: uint,
    priv computed: bool,
    priv work_buf: [u32, ..WORK_BUF_LEN]
}

fn add_input(st: &mut Sha1, msg: &[u8]) {
    assert!((!st.computed));
    for msg.iter().advance |element| {
        st.msg_block[st.msg_block_idx] = *element;
        st.msg_block_idx += 1;
        st.len_low += 8;
        if st.len_low == 0 {
            st.len_high += 1;
            if st.len_high == 0 {
                // FIXME: Need better failure mode (#2346)
                fail!();
            }
        }
        if st.msg_block_idx == MSG_BLOCK_LEN { process_msg_block(st); }
    }
}

fn process_msg_block(st: &mut Sha1) {
    let mut t: int; // Loop counter
    let mut w = st.work_buf;

    // Initialize the first 16 words of the vector w
    t = 0;
    while t < 16 {
        let mut tmp;
        tmp = (st.msg_block[t * 4] as u32) << 24u32;
        tmp = tmp | (st.msg_block[t * 4 + 1] as u32) << 16u32;
        tmp = tmp | (st.msg_block[t * 4 + 2] as u32) << 8u32;
        tmp = tmp | (st.msg_block[t * 4 + 3] as u32);
        w[t] = tmp;
        t += 1;
    }

    // Initialize the rest of vector w
    while t < 80 {
        let val = w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16];
        w[t] = circular_shift(1, val);
        t += 1;
    }
    let mut a = st.h[0];
    let mut b = st.h[1];
    let mut c = st.h[2];
    let mut d = st.h[3];
    let mut e = st.h[4];
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
    st.h[0] = st.h[0] + a;
    st.h[1] = st.h[1] + b;
    st.h[2] = st.h[2] + c;
    st.h[3] = st.h[3] + d;
    st.h[4] = st.h[4] + e;
    st.msg_block_idx = 0;
}

fn circular_shift(bits: u32, word: u32) -> u32 {
    return word << bits | word >> 32u32 - bits;
}

fn mk_result(st: &mut Sha1, rs: &mut [u8]) {
    if !st.computed { pad_msg(st); st.computed = true; }
    let mut i = 0;
    for st.h.mut_iter().advance |ptr_hpart| {
        let hpart = *ptr_hpart;
        rs[i]   = (hpart >> 24u32 & 0xFFu32) as u8;
        rs[i+1] = (hpart >> 16u32 & 0xFFu32) as u8;
        rs[i+2] = (hpart >> 8u32 & 0xFFu32) as u8;
        rs[i+3] = (hpart & 0xFFu32) as u8;
        i += 4;
    }
}

/*
 * According to the standard, the message must be padded to an even
 * 512 bits.  The first padding bit must be a '1'.  The last 64 bits
 * represent the length of the original message.  All bits in between
 * should be 0.  This function will pad the message according to those
 * rules by filling the msg_block vector accordingly.  It will also
 * call process_msg_block() appropriately.  When it returns, it
 * can be assumed that the message digest has been computed.
 */
fn pad_msg(st: &mut Sha1) {
    /*
     * Check to see if the current message block is too small to hold
     * the initial padding bits and length.  If so, we will pad the
     * block, process it, and then continue padding into a second block.
     */
    if st.msg_block_idx > 55 {
        st.msg_block[st.msg_block_idx] = 0x80;
        st.msg_block_idx += 1;
        while st.msg_block_idx < MSG_BLOCK_LEN {
            st.msg_block[st.msg_block_idx] = 0;
            st.msg_block_idx += 1;
        }
        process_msg_block(st);
    } else {
        st.msg_block[st.msg_block_idx] = 0x80;
        st.msg_block_idx += 1;
    }
    while st.msg_block_idx < 56 {
        st.msg_block[st.msg_block_idx] = 0u8;
        st.msg_block_idx += 1;
    }

    // Store the message length as the last 8 octets
    st.msg_block[56] = (st.len_high >> 24u32 & 0xFFu32) as u8;
    st.msg_block[57] = (st.len_high >> 16u32 & 0xFFu32) as u8;
    st.msg_block[58] = (st.len_high >> 8u32 & 0xFFu32) as u8;
    st.msg_block[59] = (st.len_high & 0xFFu32) as u8;
    st.msg_block[60] = (st.len_low >> 24u32 & 0xFFu32) as u8;
    st.msg_block[61] = (st.len_low >> 16u32 & 0xFFu32) as u8;
    st.msg_block[62] = (st.len_low >> 8u32 & 0xFFu32) as u8;
    st.msg_block[63] = (st.len_low & 0xFFu32) as u8;
    process_msg_block(st);
}

impl Sha1 {
    /// Construct a `sha` object
    pub fn new() -> Sha1 {
        let mut st = Sha1 {
            h: [0u32, ..DIGEST_BUF_LEN],
            len_low: 0u32,
            len_high: 0u32,
            msg_block: [0u8, ..MSG_BLOCK_LEN],
            msg_block_idx: 0,
            computed: false,
            work_buf: [0u32, ..WORK_BUF_LEN]
        };
        st.reset();
        return st;
    }
}

impl Digest for Sha1 {
    pub fn reset(&mut self) {
        self.len_low = 0;
        self.len_high = 0;
        self.msg_block_idx = 0;
        self.h[0] = 0x67452301u32;
        self.h[1] = 0xEFCDAB89u32;
        self.h[2] = 0x98BADCFEu32;
        self.h[3] = 0x10325476u32;
        self.h[4] = 0xC3D2E1F0u32;
        self.computed = false;
    }
    pub fn input(&mut self, msg: &[u8]) { add_input(self, msg); }
    pub fn result(&mut self, out: &mut [u8]) { return mk_result(self, out); }
    pub fn output_bits(&self) -> uint { 160 }
}

#[cfg(test)]
mod tests {

    use digest::{Digest, DigestUtil};
    use sha1::Sha1;

    #[deriving(Clone)]
    struct Test {
        input: ~str,
        output: ~[u8],
        output_str: ~str,
    }

    #[test]
    fn test() {
        fn a_million_letter_a() -> ~str {
            let mut i = 0;
            let mut rs = ~"";
            while i < 100000 {
                rs.push_str("aaaaaaaaaa");
                i += 1;
            }
            return rs;
        }
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
            Test {
                input: a_million_letter_a(),
                output: ~[
                    0x34u8, 0xAAu8, 0x97u8, 0x3Cu8,
                    0xD4u8, 0xC4u8, 0xDAu8, 0xA4u8,
                    0xF6u8, 0x1Eu8, 0xEBu8, 0x2Bu8,
                    0xDBu8, 0xADu8, 0x27u8, 0x31u8,
                    0x65u8, 0x34u8, 0x01u8, 0x6Fu8,
                ],
                output_str: ~"34aa973cd4c4daa4f61eeb2bdbad27316534016f"
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
        for tests.iter().advance |t| {
            (*sh).input_str(t.input);
            sh.result(out);
            assert!(t.output.as_slice() == out);

            let out_str = (*sh).result_str();
            assert_eq!(out_str.len(), 40);
            assert!(out_str == t.output_str);

            sh.reset();
        }


        // Test that it works when accepting the message in pieces
        for tests.iter().advance |t| {
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
}
