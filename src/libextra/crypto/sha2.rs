// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::uint;

use cryptoutil::{write_u64_be, write_u32_be, read_u64v_be, read_u32v_be, add_bytes_to_bits,
    add_bytes_to_bits_tuple, FixedBuffer, FixedBuffer128, FixedBuffer64, StandardPadding};
use digest::Digest;


// Sha-512 and Sha-256 use basically the same calculations which are implemented by these macros.
// Inlining the calculations seems to result in better generated code.
macro_rules! schedule_round( ($t:expr) => (
        W[$t] = sigma1(W[$t - 2]) + W[$t - 7] + sigma0(W[$t - 15]) + W[$t - 16];
    )
)

macro_rules! sha2_round(
    ($A:ident, $B:ident, $C:ident, $D:ident,
     $E:ident, $F:ident, $G:ident, $H:ident, $K:ident, $t:expr) => (
        {
            $H += sum1($E) + ch($E, $F, $G) + $K[$t] + W[$t];
            $D += $H;
            $H += sum0($A) + maj($A, $B, $C);
        }
    )
)


// A structure that represents that state of a digest computation for the SHA-2 512 family of digest
// functions
struct Engine512State {
    H0: u64,
    H1: u64,
    H2: u64,
    H3: u64,
    H4: u64,
    H5: u64,
    H6: u64,
    H7: u64,
}

impl Engine512State {
    fn new(h: &[u64, ..8]) -> Engine512State {
        return Engine512State {
            H0: h[0],
            H1: h[1],
            H2: h[2],
            H3: h[3],
            H4: h[4],
            H5: h[5],
            H6: h[6],
            H7: h[7]
        };
    }

    fn reset(&mut self, h: &[u64, ..8]) {
        self.H0 = h[0];
        self.H1 = h[1];
        self.H2 = h[2];
        self.H3 = h[3];
        self.H4 = h[4];
        self.H5 = h[5];
        self.H6 = h[6];
        self.H7 = h[7];
    }

    fn process_block(&mut self, data: &[u8]) {
        fn ch(x: u64, y: u64, z: u64) -> u64 {
            ((x & y) ^ ((!x) & z))
        }

        fn maj(x: u64, y: u64, z: u64) -> u64 {
            ((x & y) ^ (x & z) ^ (y & z))
        }

        fn sum0(x: u64) -> u64 {
            ((x << 36) | (x >> 28)) ^ ((x << 30) | (x >> 34)) ^ ((x << 25) | (x >> 39))
        }

        fn sum1(x: u64) -> u64 {
            ((x << 50) | (x >> 14)) ^ ((x << 46) | (x >> 18)) ^ ((x << 23) | (x >> 41))
        }

        fn sigma0(x: u64) -> u64 {
            ((x << 63) | (x >> 1)) ^ ((x << 56) | (x >> 8)) ^ (x >> 7)
        }

        fn sigma1(x: u64) -> u64 {
            ((x << 45) | (x >> 19)) ^ ((x << 3) | (x >> 61)) ^ (x >> 6)
        }

        let mut a = self.H0;
        let mut b = self.H1;
        let mut c = self.H2;
        let mut d = self.H3;
        let mut e = self.H4;
        let mut f = self.H5;
        let mut g = self.H6;
        let mut h = self.H7;

        let mut W = [0u64, ..80];

        read_u64v_be(W.mut_slice(0, 16), data);

        // Putting the message schedule inside the same loop as the round calculations allows for
        // the compiler to generate better code.
        do uint::range_step(0, 64, 8) |t| {
            schedule_round!(t + 16);
            schedule_round!(t + 17);
            schedule_round!(t + 18);
            schedule_round!(t + 19);
            schedule_round!(t + 20);
            schedule_round!(t + 21);
            schedule_round!(t + 22);
            schedule_round!(t + 23);

            sha2_round!(a, b, c, d, e, f, g, h, K64, t);
            sha2_round!(h, a, b, c, d, e, f, g, K64, t + 1);
            sha2_round!(g, h, a, b, c, d, e, f, K64, t + 2);
            sha2_round!(f, g, h, a, b, c, d, e, K64, t + 3);
            sha2_round!(e, f, g, h, a, b, c, d, K64, t + 4);
            sha2_round!(d, e, f, g, h, a, b, c, K64, t + 5);
            sha2_round!(c, d, e, f, g, h, a, b, K64, t + 6);
            sha2_round!(b, c, d, e, f, g, h, a, K64, t + 7);
            true
        };

        do uint::range_step(64, 80, 8) |t| {
            sha2_round!(a, b, c, d, e, f, g, h, K64, t);
            sha2_round!(h, a, b, c, d, e, f, g, K64, t + 1);
            sha2_round!(g, h, a, b, c, d, e, f, K64, t + 2);
            sha2_round!(f, g, h, a, b, c, d, e, K64, t + 3);
            sha2_round!(e, f, g, h, a, b, c, d, K64, t + 4);
            sha2_round!(d, e, f, g, h, a, b, c, K64, t + 5);
            sha2_round!(c, d, e, f, g, h, a, b, K64, t + 6);
            sha2_round!(b, c, d, e, f, g, h, a, K64, t + 7);
            true
        };

        self.H0 += a;
        self.H1 += b;
        self.H2 += c;
        self.H3 += d;
        self.H4 += e;
        self.H5 += f;
        self.H6 += g;
        self.H7 += h;
    }
}

// Constants necessary for SHA-2 512 family of digests.
static K64: [u64, ..80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817
];


// A structure that keeps track of the state of the Sha-512 operation and contains the logic
// necessary to perform the final calculations.
struct Engine512 {
    length_bits: (u64, u64),
    buffer: FixedBuffer128,
    state: Engine512State,
    finished: bool,
}

impl Engine512 {
    fn new(h: &[u64, ..8]) -> Engine512 {
        return Engine512 {
            length_bits: (0, 0),
            buffer: FixedBuffer128::new(),
            state: Engine512State::new(h),
            finished: false
        }
    }

    fn reset(&mut self, h: &[u64, ..8]) {
        self.length_bits = (0, 0);
        self.buffer.reset();
        self.state.reset(h);
        self.finished = false;
    }

    fn input(&mut self, input: &[u8]) {
        assert!(!self.finished)
        // Assumes that input.len() can be converted to u64 without overflow
        self.length_bits = add_bytes_to_bits_tuple(self.length_bits, input.len() as u64);
        self.buffer.input(input, |input: &[u8]| { self.state.process_block(input) });
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }

        self.buffer.standard_padding(16, |input: &[u8]| { self.state.process_block(input) });
        match self.length_bits {
            (hi, low) => {
                write_u64_be(self.buffer.next(8), hi);
                write_u64_be(self.buffer.next(8), low);
            }
        }
        self.state.process_block(self.buffer.full_buffer());

        self.finished = true;
    }
}


struct Sha512 {
    priv engine: Engine512
}

impl Sha512 {
    /**
     * Construct an new instance of a SHA-512 digest.
     */
    pub fn new() -> Sha512 {
        return Sha512 {
            engine: Engine512::new(&H512)
        };
    }
}

impl Digest for Sha512 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.finish();

        write_u64_be(out.mut_slice(0, 8), self.engine.state.H0);
        write_u64_be(out.mut_slice(8, 16), self.engine.state.H1);
        write_u64_be(out.mut_slice(16, 24), self.engine.state.H2);
        write_u64_be(out.mut_slice(24, 32), self.engine.state.H3);
        write_u64_be(out.mut_slice(32, 40), self.engine.state.H4);
        write_u64_be(out.mut_slice(40, 48), self.engine.state.H5);
        write_u64_be(out.mut_slice(48, 56), self.engine.state.H6);
        write_u64_be(out.mut_slice(56, 64), self.engine.state.H7);
    }

    fn reset(&mut self) {
        self.engine.reset(&H512);
    }

    fn output_bits(&self) -> uint { 512 }
}

static H512: [u64, ..8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179
];


struct Sha384 {
    priv engine: Engine512
}

impl Sha384 {
    /**
     * Construct an new instance of a SHA-384 digest.
     */
    pub fn new() -> Sha384 {
        Sha384 {
            engine: Engine512::new(&H384)
        }
    }
}

impl Digest for Sha384 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.finish();

        write_u64_be(out.mut_slice(0, 8), self.engine.state.H0);
        write_u64_be(out.mut_slice(8, 16), self.engine.state.H1);
        write_u64_be(out.mut_slice(16, 24), self.engine.state.H2);
        write_u64_be(out.mut_slice(24, 32), self.engine.state.H3);
        write_u64_be(out.mut_slice(32, 40), self.engine.state.H4);
        write_u64_be(out.mut_slice(40, 48), self.engine.state.H5);
    }

    fn reset(&mut self) {
        self.engine.reset(&H384);
    }

    fn output_bits(&self) -> uint { 384 }
}

static H384: [u64, ..8] = [
    0xcbbb9d5dc1059ed8,
    0x629a292a367cd507,
    0x9159015a3070dd17,
    0x152fecd8f70e5939,
    0x67332667ffc00b31,
    0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7,
    0x47b5481dbefa4fa4
];


struct Sha512Trunc256 {
    priv engine: Engine512
}

impl Sha512Trunc256 {
    /**
     * Construct an new instance of a SHA-512/256 digest.
     */
    pub fn new() -> Sha512Trunc256 {
        Sha512Trunc256 {
            engine: Engine512::new(&H512_TRUNC_256)
        }
    }
}

impl Digest for Sha512Trunc256 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.finish();

        write_u64_be(out.mut_slice(0, 8), self.engine.state.H0);
        write_u64_be(out.mut_slice(8, 16), self.engine.state.H1);
        write_u64_be(out.mut_slice(16, 24), self.engine.state.H2);
        write_u64_be(out.mut_slice(24, 32), self.engine.state.H3);
    }

    fn reset(&mut self) {
        self.engine.reset(&H512_TRUNC_256);
    }

    fn output_bits(&self) -> uint { 256 }
}

static H512_TRUNC_256: [u64, ..8] = [
    0x22312194fc2bf72c,
    0x9f555fa3c84c64c2,
    0x2393b86b6f53b151,
    0x963877195940eabd,
    0x96283ee2a88effe3,
    0xbe5e1e2553863992,
    0x2b0199fc2c85b8aa,
    0x0eb72ddc81c52ca2
];


struct Sha512Trunc224 {
    priv engine: Engine512
}

impl Sha512Trunc224 {
    /**
     * Construct an new instance of a SHA-512/224 digest.
     */
    pub fn new() -> Sha512Trunc224 {
        Sha512Trunc224 {
            engine: Engine512::new(&H512_TRUNC_224)
        }
    }
}

impl Digest for Sha512Trunc224 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.finish();

        write_u64_be(out.mut_slice(0, 8), self.engine.state.H0);
        write_u64_be(out.mut_slice(8, 16), self.engine.state.H1);
        write_u64_be(out.mut_slice(16, 24), self.engine.state.H2);
        write_u32_be(out.mut_slice(24, 28), (self.engine.state.H3 >> 32) as u32);
    }

    fn reset(&mut self) {
        self.engine.reset(&H512_TRUNC_224);
    }

    fn output_bits(&self) -> uint { 224 }
}

static H512_TRUNC_224: [u64, ..8] = [
    0x8c3d37c819544da2,
    0x73e1996689dcd4d6,
    0x1dfab7ae32ff9c82,
    0x679dd514582f9fcf,
    0x0f6d2b697bd44da8,
    0x77e36f7304c48942,
    0x3f9d85a86a1d36c8,
    0x1112e6ad91d692a1,
];


// A structure that represents that state of a digest computation for the SHA-2 512 family of digest
// functions
struct Engine256State {
    H0: u32,
    H1: u32,
    H2: u32,
    H3: u32,
    H4: u32,
    H5: u32,
    H6: u32,
    H7: u32,
}

impl Engine256State {
    fn new(h: &[u32, ..8]) -> Engine256State {
        return Engine256State {
            H0: h[0],
            H1: h[1],
            H2: h[2],
            H3: h[3],
            H4: h[4],
            H5: h[5],
            H6: h[6],
            H7: h[7]
        };
    }

    fn reset(&mut self, h: &[u32, ..8]) {
        self.H0 = h[0];
        self.H1 = h[1];
        self.H2 = h[2];
        self.H3 = h[3];
        self.H4 = h[4];
        self.H5 = h[5];
        self.H6 = h[6];
        self.H7 = h[7];
    }

    fn process_block(&mut self, data: &[u8]) {
        fn ch(x: u32, y: u32, z: u32) -> u32 {
            ((x & y) ^ ((!x) & z))
        }

        fn maj(x: u32, y: u32, z: u32) -> u32 {
            ((x & y) ^ (x & z) ^ (y & z))
        }

        fn sum0(x: u32) -> u32 {
            ((x >> 2) | (x << 30)) ^ ((x >> 13) | (x << 19)) ^ ((x >> 22) | (x << 10))
        }

        fn sum1(x: u32) -> u32 {
            ((x >> 6) | (x << 26)) ^ ((x >> 11) | (x << 21)) ^ ((x >> 25) | (x << 7))
        }

        fn sigma0(x: u32) -> u32 {
            ((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3)
        }

        fn sigma1(x: u32) -> u32 {
            ((x >> 17) | (x << 15)) ^ ((x >> 19) | (x << 13)) ^ (x >> 10)
        }

        let mut a = self.H0;
        let mut b = self.H1;
        let mut c = self.H2;
        let mut d = self.H3;
        let mut e = self.H4;
        let mut f = self.H5;
        let mut g = self.H6;
        let mut h = self.H7;

        let mut W = [0u32, ..64];

        read_u32v_be(W.mut_slice(0, 16), data);

        // Putting the message schedule inside the same loop as the round calculations allows for
        // the compiler to generate better code.
        do uint::range_step(0, 48, 8) |t| {
            schedule_round!(t + 16);
            schedule_round!(t + 17);
            schedule_round!(t + 18);
            schedule_round!(t + 19);
            schedule_round!(t + 20);
            schedule_round!(t + 21);
            schedule_round!(t + 22);
            schedule_round!(t + 23);

            sha2_round!(a, b, c, d, e, f, g, h, K32, t);
            sha2_round!(h, a, b, c, d, e, f, g, K32, t + 1);
            sha2_round!(g, h, a, b, c, d, e, f, K32, t + 2);
            sha2_round!(f, g, h, a, b, c, d, e, K32, t + 3);
            sha2_round!(e, f, g, h, a, b, c, d, K32, t + 4);
            sha2_round!(d, e, f, g, h, a, b, c, K32, t + 5);
            sha2_round!(c, d, e, f, g, h, a, b, K32, t + 6);
            sha2_round!(b, c, d, e, f, g, h, a, K32, t + 7);
            true
        };

        do uint::range_step(48, 64, 8) |t| {
            sha2_round!(a, b, c, d, e, f, g, h, K32, t);
            sha2_round!(h, a, b, c, d, e, f, g, K32, t + 1);
            sha2_round!(g, h, a, b, c, d, e, f, K32, t + 2);
            sha2_round!(f, g, h, a, b, c, d, e, K32, t + 3);
            sha2_round!(e, f, g, h, a, b, c, d, K32, t + 4);
            sha2_round!(d, e, f, g, h, a, b, c, K32, t + 5);
            sha2_round!(c, d, e, f, g, h, a, b, K32, t + 6);
            sha2_round!(b, c, d, e, f, g, h, a, K32, t + 7);
            true
        };

        self.H0 += a;
        self.H1 += b;
        self.H2 += c;
        self.H3 += d;
        self.H4 += e;
        self.H5 += f;
        self.H6 += g;
        self.H7 += h;
    }
}

static K32: [u32, ..64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
];


// A structure that keeps track of the state of the Sha-256 operation and contains the logic
// necessary to perform the final calculations.
struct Engine256 {
    length_bits: u64,
    buffer: FixedBuffer64,
    state: Engine256State,
    finished: bool,
}

impl Engine256 {
    fn new(h: &[u32, ..8]) -> Engine256 {
        return Engine256 {
            length_bits: 0,
            buffer: FixedBuffer64::new(),
            state: Engine256State::new(h),
            finished: false
        }
    }

    fn reset(&mut self, h: &[u32, ..8]) {
        self.length_bits = 0;
        self.buffer.reset();
        self.state.reset(h);
        self.finished = false;
    }

    fn input(&mut self, input: &[u8]) {
        assert!(!self.finished)
        // Assumes that input.len() can be converted to u64 without overflow
        self.length_bits = add_bytes_to_bits(self.length_bits, input.len() as u64);
        self.buffer.input(input, |input: &[u8]| { self.state.process_block(input) });
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }

        self.buffer.standard_padding(8, |input: &[u8]| { self.state.process_block(input) });
        write_u32_be(self.buffer.next(4), (self.length_bits >> 32) as u32 );
        write_u32_be(self.buffer.next(4), self.length_bits as u32);
        self.state.process_block(self.buffer.full_buffer());

        self.finished = true;
    }
}


struct Sha256 {
    priv engine: Engine256
}

impl Sha256 {
    /**
     * Construct an new instance of a SHA-256 digest.
     */
    pub fn new() -> Sha256 {
        Sha256 {
            engine: Engine256::new(&H256)
        }
    }
}

impl Digest for Sha256 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.finish();

        write_u32_be(out.mut_slice(0, 4), self.engine.state.H0);
        write_u32_be(out.mut_slice(4, 8), self.engine.state.H1);
        write_u32_be(out.mut_slice(8, 12), self.engine.state.H2);
        write_u32_be(out.mut_slice(12, 16), self.engine.state.H3);
        write_u32_be(out.mut_slice(16, 20), self.engine.state.H4);
        write_u32_be(out.mut_slice(20, 24), self.engine.state.H5);
        write_u32_be(out.mut_slice(24, 28), self.engine.state.H6);
        write_u32_be(out.mut_slice(28, 32), self.engine.state.H7);
    }

    fn reset(&mut self) {
        self.engine.reset(&H256);
    }

    fn output_bits(&self) -> uint { 256 }
}

static H256: [u32, ..8] = [
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19
];


struct Sha224 {
    priv engine: Engine256
}

impl Sha224 {
    /**
     * Construct an new instance of a SHA-224 digest.
     */
    pub fn new() -> Sha224 {
        Sha224 {
            engine: Engine256::new(&H224)
        }
    }
}

impl Digest for Sha224 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.finish();
        write_u32_be(out.mut_slice(0, 4), self.engine.state.H0);
        write_u32_be(out.mut_slice(4, 8), self.engine.state.H1);
        write_u32_be(out.mut_slice(8, 12), self.engine.state.H2);
        write_u32_be(out.mut_slice(12, 16), self.engine.state.H3);
        write_u32_be(out.mut_slice(16, 20), self.engine.state.H4);
        write_u32_be(out.mut_slice(20, 24), self.engine.state.H5);
        write_u32_be(out.mut_slice(24, 28), self.engine.state.H6);
    }

    fn reset(&mut self) {
        self.engine.reset(&H224);
    }

    fn output_bits(&self) -> uint { 224 }
}

static H224: [u32, ..8] = [
    0xc1059ed8,
    0x367cd507,
    0x3070dd17,
    0xf70e5939,
    0xffc00b31,
    0x68581511,
    0x64f98fa7,
    0xbefa4fa4
];


#[cfg(test)]
mod tests {
    use cryptoutil::test::test_digest_1million_random;
    use digest::Digest;
    use sha2::{Sha512, Sha384, Sha512Trunc256, Sha512Trunc224, Sha256, Sha224};

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
    fn test_sha512() {
        // Examples from wikipedia
        let wikipedia_tests = ~[
            Test {
                input: ~"",
                output_str: ~"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce" +
                             "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog",
                output_str: ~"07e547d9586f6a73f73fbac0435ed76951218fb7d0c8d788a309d785436bbb64" +
                             "2e93a252a954f23912547d1e8a3b5ed6e1bfd7097821233fa0538f3db854fee6"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog.",
                output_str: ~"91ea1245f20d46ae9a037a989f54f1f790f0a47607eeb8a14d12890cea77a1bb" +
                             "c6c7ed9cf205e67b7f2b8fd4c7dfd3a7a8617e45f3c463d481c7e586c39ac1ed"
            },
        ];

        let tests = wikipedia_tests;

        let mut sh = ~Sha512::new();

        test_hash(sh, tests);
    }

    #[test]
    fn test_sha384() {
        // Examples from wikipedia
        let wikipedia_tests = ~[
            Test {
                input: ~"",
                output_str: ~"38b060a751ac96384cd9327eb1b1e36a21fdb71114be0743" +
                             "4c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog",
                output_str: ~"ca737f1014a48f4c0b6dd43cb177b0afd9e5169367544c49" +
                             "4011e3317dbf9a509cb1e5dc1e85a941bbee3d7f2afbc9b1"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog.",
                output_str: ~"ed892481d8272ca6df370bf706e4d7bc1b5739fa2177aae6" +
                             "c50e946678718fc67a7af2819a021c2fc34e91bdb63409d7"
            },
        ];

        let tests = wikipedia_tests;

        let mut sh = ~Sha384::new();

        test_hash(sh, tests);
    }

    #[test]
    fn test_sha512_256() {
        // Examples from wikipedia
        let wikipedia_tests = ~[
            Test {
                input: ~"",
                output_str: ~"c672b8d1ef56ed28ab87c3622c5114069bdd3ad7b8f9737498d0c01ecef0967a"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog",
                output_str: ~"dd9d67b371519c339ed8dbd25af90e976a1eeefd4ad3d889005e532fc5bef04d"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog.",
                output_str: ~"1546741840f8a492b959d9b8b2344b9b0eb51b004bba35c0aebaac86d45264c3"
            },
        ];

        let tests = wikipedia_tests;

        let mut sh = ~Sha512Trunc256::new();

        test_hash(sh, tests);
    }

    #[test]
    fn test_sha512_224() {
        // Examples from wikipedia
        let wikipedia_tests = ~[
            Test {
                input: ~"",
                output_str: ~"6ed0dd02806fa89e25de060c19d3ac86cabb87d6a0ddd05c333b84f4"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog",
                output_str: ~"944cd2847fb54558d4775db0485a50003111c8e5daa63fe722c6aa37"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog.",
                output_str: ~"6d6a9279495ec4061769752e7ff9c68b6b0b3c5a281b7917ce0572de"
            },
        ];

        let tests = wikipedia_tests;

        let mut sh = ~Sha512Trunc224::new();

        test_hash(sh, tests);
    }

    #[test]
    fn test_sha256() {
        // Examples from wikipedia
        let wikipedia_tests = ~[
            Test {
                input: ~"",
                output_str: ~"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog",
                output_str: ~"d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog.",
                output_str: ~"ef537f25c895bfa782526529a9b63d97aa631564d5d789c2b765448c8635fb6c"
            },
        ];

        let tests = wikipedia_tests;

        let mut sh = ~Sha256::new();

        test_hash(sh, tests);
    }

    #[test]
    fn test_sha224() {
        // Examples from wikipedia
        let wikipedia_tests = ~[
            Test {
                input: ~"",
                output_str: ~"d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog",
                output_str: ~"730e109bd7a8a32b1cb9d9a09aa2325d2430587ddbc0c38bad911525"
            },
            Test {
                input: ~"The quick brown fox jumps over the lazy dog.",
                output_str: ~"619cba8e8e05826e9b8c519c0a5c68f4fb653e8a3d8aa04bb2c8cd4c"
            },
        ];

        let tests = wikipedia_tests;

        let mut sh = ~Sha224::new();

        test_hash(sh, tests);
    }

    #[test]
    fn test_1million_random_sha512() {
        let mut sh = Sha512::new();
        test_digest_1million_random(
            &mut sh,
            128,
            "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973eb" +
            "de0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b");
        }

    #[test]
    fn test_1million_random_sha256() {
        let mut sh = Sha256::new();
        test_digest_1million_random(
            &mut sh,
            64,
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }
}



#[cfg(test)]
mod bench {

    use sha2::{Sha256,Sha512};
    use test::BenchHarness;

    #[bench]
    pub fn sha256_10(bh: & mut BenchHarness) {
        let mut sh = Sha256::new();
        let bytes = [1u8, ..10];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn sha256_1k(bh: & mut BenchHarness) {
        let mut sh = Sha256::new();
        let bytes = [1u8, ..1024];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn sha256_64k(bh: & mut BenchHarness) {
        let mut sh = Sha256::new();
        let bytes = [1u8, ..65536];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }



    #[bench]
    pub fn sha512_10(bh: & mut BenchHarness) {
        let mut sh = Sha512::new();
        let bytes = [1u8, ..10];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn sha512_1k(bh: & mut BenchHarness) {
        let mut sh = Sha512::new();
        let bytes = [1u8, ..1024];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn sha512_64k(bh: & mut BenchHarness) {
        let mut sh = Sha512::new();
        let bytes = [1u8, ..65536];
        do bh.iter {
            sh.input(bytes);
        }
        bh.bytes = bytes.len() as u64;
    }

}
