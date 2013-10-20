// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

use std::iter::range_step;

use cryptoutil::{write_u32_le, read_u32v_le, FixedBuffer, FixedBuffer64, StandardPadding};
use digest::Digest;


// A structure that represents that state of a digest computation for the MD5 digest function
struct Md5State {
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32
}

impl Md5State {
    fn new() -> Md5State {
        return Md5State {
            s0: 0x67452301,
            s1: 0xefcdab89,
            s2: 0x98badcfe,
            s3: 0x10325476
        };
    }

    fn reset(&mut self) {
        self.s0 = 0x67452301;
        self.s1 = 0xefcdab89;
        self.s2 = 0x98badcfe;
        self.s3 = 0x10325476;
    }

    fn process_block(&mut self, input: &[u8]) {
        fn f(u: u32, v: u32, w: u32) -> u32 {
            return (u & v) | (!u & w);
        }

        fn g(u: u32, v: u32, w: u32) -> u32 {
            return (u & w) | (v & !w);
        }

        fn h(u: u32, v: u32, w: u32) -> u32 {
            return u ^ v ^ w;
        }

        fn i(u: u32, v: u32, w: u32) -> u32 {
            return v ^ (u | !w);
        }

        fn rotate_left(x: u32, n: u32) -> u32 {
            return (x << n) | (x >> (32 - n));
        }

        fn op_f(w: u32, x: u32, y: u32, z: u32, m: u32, s: u32) -> u32 {
            return rotate_left(w + f(x, y, z) + m, s) + x;
        }

        fn op_g(w: u32, x: u32, y: u32, z: u32, m: u32, s: u32) -> u32 {
            return rotate_left(w + g(x, y, z) + m, s) + x;
        }

        fn op_h(w: u32, x: u32, y: u32, z: u32, m: u32, s: u32) -> u32 {
            return rotate_left(w + h(x, y, z) + m, s) + x;
        }

        fn op_i(w: u32, x: u32, y: u32, z: u32, m: u32, s: u32) -> u32 {
            return rotate_left(w + i(x, y, z) + m, s) + x;
        }

        let mut a = self.s0;
        let mut b = self.s1;
        let mut c = self.s2;
        let mut d = self.s3;

        let mut data = [0u32, ..16];

        read_u32v_le(data, input);

        // round 1
        for i in range_step(0u, 16, 4) {
            a = op_f(a, b, c, d, data[i] + C1[i], 7);
            d = op_f(d, a, b, c, data[i + 1] + C1[i + 1], 12);
            c = op_f(c, d, a, b, data[i + 2] + C1[i + 2], 17);
            b = op_f(b, c, d, a, data[i + 3] + C1[i + 3], 22);
        }

        // round 2
        let mut t = 1;
        for i in range_step(0u, 16, 4) {
            a = op_g(a, b, c, d, data[t & 0x0f] + C2[i], 5);
            d = op_g(d, a, b, c, data[(t + 5) & 0x0f] + C2[i + 1], 9);
            c = op_g(c, d, a, b, data[(t + 10) & 0x0f] + C2[i + 2], 14);
            b = op_g(b, c, d, a, data[(t + 15) & 0x0f] + C2[i + 3], 20);
            t += 20;
        }

        // round 3
        t = 5;
        for i in range_step(0u, 16, 4) {
            a = op_h(a, b, c, d, data[t & 0x0f] + C3[i], 4);
            d = op_h(d, a, b, c, data[(t + 3) & 0x0f] + C3[i + 1], 11);
            c = op_h(c, d, a, b, data[(t + 6) & 0x0f] + C3[i + 2], 16);
            b = op_h(b, c, d, a, data[(t + 9) & 0x0f] + C3[i + 3], 23);
            t += 12;
        }

        // round 4
        t = 0;
        for i in range_step(0u, 16, 4) {
            a = op_i(a, b, c, d, data[t & 0x0f] + C4[i], 6);
            d = op_i(d, a, b, c, data[(t + 7) & 0x0f] + C4[i + 1], 10);
            c = op_i(c, d, a, b, data[(t + 14) & 0x0f] + C4[i + 2], 15);
            b = op_i(b, c, d, a, data[(t + 21) & 0x0f] + C4[i + 3], 21);
            t += 28;
        }

        self.s0 += a;
        self.s1 += b;
        self.s2 += c;
        self.s3 += d;
    }
}

// Round 1 constants
static C1: [u32, ..16] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821
];

// Round 2 constants
static C2: [u32, ..16] = [
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a
];

// Round 3 constants
static C3: [u32, ..16] = [
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665
];

// Round 4 constants
static C4: [u32, ..16] = [
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
];


/// The MD5 Digest algorithm
pub struct Md5 {
    priv length_bytes: u64,
    priv buffer: FixedBuffer64,
    priv state: Md5State,
    priv finished: bool,
}

impl Md5 {
    /// Construct a new instance of the MD5 Digest.
    pub fn new() -> Md5 {
        return Md5 {
            length_bytes: 0,
            buffer: FixedBuffer64::new(),
            state: Md5State::new(),
            finished: false
        }
    }
}

impl Digest for Md5 {
    fn input(&mut self, input: &[u8]) {
        assert!(!self.finished);
        // Unlike Sha1 and Sha2, the length value in MD5 is defined as the length of the message mod
        // 2^64 - ie: integer overflow is OK.
        self.length_bytes += input.len() as u64;
        self.buffer.input(input, |d: &[u8]| { self.state.process_block(d); });
    }

    fn reset(&mut self) {
        self.length_bytes = 0;
        self.buffer.reset();
        self.state.reset();
        self.finished = false;
    }

    fn result(&mut self, out: &mut [u8]) {
        if !self.finished {
            self.buffer.standard_padding(8, |d: &[u8]| { self.state.process_block(d); });
            write_u32_le(self.buffer.next(4), (self.length_bytes << 3) as u32);
            write_u32_le(self.buffer.next(4), (self.length_bytes >> 29) as u32);
            self.state.process_block(self.buffer.full_buffer());
            self.finished = true;
        }

        write_u32_le(out.mut_slice(0, 4), self.state.s0);
        write_u32_le(out.mut_slice(4, 8), self.state.s1);
        write_u32_le(out.mut_slice(8, 12), self.state.s2);
        write_u32_le(out.mut_slice(12, 16), self.state.s3);
    }

    fn output_bits(&self) -> uint { 128 }
}


#[cfg(test)]
mod tests {
    use cryptoutil::test::test_digest_1million_random;
    use digest::Digest;
    use md5::Md5;


    struct Test {
        input: ~str,
        output_str: ~str,
    }

    fn test_hash<D: Digest>(sh: &mut D, tests: &[Test]) {
        // Test that it works when accepting the message all at once
        for t in tests.iter() {
            sh.input_str(t.input);

            let out_str = sh.result_str();
            assert!(out_str == t.output_str);

            sh.reset();
        }

        // Test that it works when accepting the message in pieces
        for t in tests.iter() {
            let len = t.input.len();
            let mut left = len;
            while left > 0u {
                let take = (left + 1u) / 2u;
                sh.input_str(t.input.slice(len - left, take + len - left));
                left = left - take;
            }

            let out_str = sh.result_str();
            assert!(out_str == t.output_str);

            sh.reset();
        }
    }

    #[test]
    fn test_md5() {
        // Examples from wikipedia
        let wikipedia_tests = ~[
            Test {
                input: ~"",
                output_str: ~"d41d8cd98f00b204e9800998ecf8427e"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog",
                output_str: ~"9e107d9d372bb6826bd81d3542a419d6"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog.",
                output_str: ~"e4d909c290d0fb1ca068ffaddf22cbd0"
            },
        ];

        let tests = wikipedia_tests;

        let mut sh = Md5::new();

        test_hash(&mut sh, tests);
    }

    #[test]
    fn test_1million_random_md5() {
        let mut sh = Md5::new();
        test_digest_1million_random(
            &mut sh,
            64,
            "7707d6ae4e027c70eea2a935c2296f21");
    }
}


#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;

    use md5::Md5;


    #[bench]
    pub fn md5_10(bh: & mut BenchHarness) {
        let mut sh = Md5::new();
        let bytes = [1u8, ..10];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn md5_1k(bh: & mut BenchHarness) {
        let mut sh = Md5::new();
        let bytes = [1u8, ..1024];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn md5_64k(bh: & mut BenchHarness) {
        let mut sh = Md5::new();
        let bytes = [1u8, ..65536];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }
}
