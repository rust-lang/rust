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

use digest::Digest;

// BitCounter is a specialized structure intended simply for counting the
// number of bits that have been processed by the SHA-2 512 family of functions.
// It does very little overflow checking since such checking is not necessary
// for how it is used. A more generic structure would have to do this checking.
// So, don't copy this structure and use it elsewhere!
struct BitCounter {
    high_bit_count: u64,
    low_byte_count: u64
}

impl BitCounter {
    fn add_bytes(&mut self, bytes: uint) {
        self.low_byte_count += bytes as u64;
        if(self.low_byte_count > 0x1fffffffffffffffu64) {
            self.high_bit_count += (self.low_byte_count >> 61);
            self.low_byte_count &= 0x1fffffffffffffffu64;
        }
    }

    fn reset(&mut self) {
        self.low_byte_count = 0;
        self.high_bit_count = 0;
    }

    fn get_low_bit_count(&self) -> u64 {
        self.low_byte_count << 3
    }

    fn get_high_bit_count(&self) -> u64 {
        self.high_bit_count
    }
}

// A structure that represents that state of a digest computation
// for the SHA-2 512 family of digest functions
struct Engine512 {
    input_buffer: [u8, ..8],
    input_buffer_idx: uint,
    bit_counter: BitCounter,
    H0: u64,
    H1: u64,
    H2: u64,
    H3: u64,
    H4: u64,
    H5: u64,
    H6: u64,
    H7: u64,
    W: [u64, ..80],
    W_idx: uint,
    finished: bool,
}

// Convert a [u8] to a u64 in big-endian format
fn to_u64(in: &[u8]) -> u64 {
    (in[0] as u64) << 56 |
    (in[1] as u64) << 48 |
    (in[2] as u64) << 40 |
    (in[3] as u64) << 32 |
    (in[4] as u64) << 24 |
    (in[5] as u64) << 16 |
    (in[6] as u64) << 8 |
    (in[7] as u64)
}

// Convert a u64 to a [u8] in big endian format
fn from_u64(in: u64, out: &mut [u8]) {
    out[0] = (in >> 56) as u8;
    out[1] = (in >> 48) as u8;
    out[2] = (in >> 40) as u8;
    out[3] = (in >> 32) as u8;
    out[4] = (in >> 24) as u8;
    out[5] = (in >> 16) as u8;
    out[6] = (in >> 8) as u8;
    out[7] = in as u8;
}

impl Engine512 {
    fn input_byte(&mut self, in: u8) {
        assert!(!self.finished)

        self.input_buffer[self.input_buffer_idx] = in;
        self.input_buffer_idx += 1;

        if (self.input_buffer_idx == 8) {
            self.input_buffer_idx = 0;
            let w = to_u64(self.input_buffer);
            self.process_word(w);
        }

        self.bit_counter.add_bytes(1);
    }

    fn input_vec(&mut self, in: &[u8]) {
        assert!(!self.finished)

        let mut i = 0;

        while i < in.len() && self.input_buffer_idx != 0 {
            self.input_byte(in[i]);
            i += 1;
        }

        while in.len() - i >= 8 {
            let w = to_u64(in.slice(i, i + 8));
            self.process_word(w);
            self.bit_counter.add_bytes(8);
            i += 8;
        }

        while i < in.len() {
            self.input_byte(in[i]);
            i += 1;
        }
    }

    fn reset(&mut self) {
        self.bit_counter.reset();
        self.finished = false;
        self.input_buffer_idx = 0;
        self.W_idx = 0;
    }

    fn process_word(&mut self, in: u64) {
        self.W[self.W_idx] = in;
        self.W_idx += 1;
        if (self.W_idx == 16) {
            self.W_idx = 0;
            self.process_block();
        }
    }

    fn process_block(&mut self) {
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

        for uint::range(16, 80) |t| {
            self.W[t] = sigma1(self.W[t - 2]) + self.W[t - 7] + sigma0(self.W[t - 15]) +
                self.W[t - 16];
        }

        let mut a = self.H0;
        let mut b = self.H1;
        let mut c = self.H2;
        let mut d = self.H3;
        let mut e = self.H4;
        let mut f = self.H5;
        let mut g = self.H6;
        let mut h = self.H7;

        let mut t = 0;
        for uint::range(0, 10) |_| {
            h += sum1(e) + ch(e, f, g) + K64[t] + self.W[t];
            d += h;
            h += sum0(a) + maj(a, b, c);
            t += 1;

            g += sum1(d) + ch(d, e, f) + K64[t] + self.W[t];
            c += g;
            g += sum0(h) + maj(h, a, b);
            t += 1;

            f += sum1(c) + ch(c, d, e) + K64[t] + self.W[t];
            b += f;
            f += sum0(g) + maj(g, h, a);
            t += 1;

            e += sum1(b) + ch(b, c, d) + K64[t] + self.W[t];
            a += e;
            e += sum0(f) + maj(f, g, h);
            t += 1;

            d += sum1(a) + ch(a, b, c) + K64[t] + self.W[t];
            h += d;
            d += sum0(e) + maj(e, f, g);
            t += 1;

            c += sum1(h) + ch(h, a, b) + K64[t] + self.W[t];
            g += c;
            c += sum0(d) + maj(d, e, f);
            t += 1;

            b += sum1(g) + ch(g, h, a) + K64[t] + self.W[t];
            f += b;
            b += sum0(c) + maj(c, d, e);
            t += 1;

            a += sum1(f) + ch(f, g, h) + K64[t] + self.W[t];
            e += a;
            a += sum0(b) + maj(b, c, d);
            t += 1;
        }

        self.H0 += a;
        self.H1 += b;
        self.H2 += c;
        self.H3 += d;
        self.H4 += e;
        self.H5 += f;
        self.H6 += g;
        self.H7 += h;
    }

    fn finish(&mut self) {
        if (self.finished) {
            return;
        }

        // must get message length before padding is added
        let high_bit_count = self.bit_counter.get_high_bit_count();
        let low_bit_count = self.bit_counter.get_low_bit_count();

        // add padding
        self.input_byte(128u8);

        while self.input_buffer_idx != 0 {
            self.input_byte(0u8);
        }

        // add length
        if (self.W_idx > 14) {
            for uint::range(self.W_idx, 16) |_| {
                self.process_word(0);
            }
        }

        while self.W_idx < 14 {
            self.process_word(0);
        }

        self.process_word(high_bit_count);
        self.process_word(low_bit_count);

        self.finished = true;
    }

    fn result_512(&mut self, out: &mut [u8]) {
        self.finish();

        from_u64(self.H0, out.mut_slice(0, 8));
        from_u64(self.H1, out.mut_slice(8, 16));
        from_u64(self.H2, out.mut_slice(16, 24));
        from_u64(self.H3, out.mut_slice(24, 32));
        from_u64(self.H4, out.mut_slice(32, 40));
        from_u64(self.H5, out.mut_slice(40, 48));
        from_u64(self.H6, out.mut_slice(48, 56));
        from_u64(self.H7, out.mut_slice(56, 64));
    }

    fn result_384(&mut self, out: &mut [u8]) {
        self.finish();

        from_u64(self.H0, out.mut_slice(0, 8));
        from_u64(self.H1, out.mut_slice(8, 16));
        from_u64(self.H2, out.mut_slice(16, 24));
        from_u64(self.H3, out.mut_slice(24, 32));
        from_u64(self.H4, out.mut_slice(32, 40));
        from_u64(self.H5, out.mut_slice(40, 48));
    }

    fn result_256(&mut self, out: &mut [u8]) {
        self.finish();

        from_u64(self.H0, out.mut_slice(0, 8));
        from_u64(self.H1, out.mut_slice(8, 16));
        from_u64(self.H2, out.mut_slice(16, 24));
        from_u64(self.H3, out.mut_slice(24, 32));
    }

    fn result_224(&mut self, out: &mut [u8]) {
        self.finish();

        from_u64(self.H0, out.mut_slice(0, 8));
        from_u64(self.H1, out.mut_slice(8, 16));
        from_u64(self.H2, out.mut_slice(16, 24));
        from_u32((self.H3 >> 32) as u32, out.mut_slice(24, 28));
    }
}

// Constants necessary for SHA-2 512 family of digests.
static K64: [u64, ..80] = [
    0x428a2f98d728ae22u64, 0x7137449123ef65cdu64, 0xb5c0fbcfec4d3b2fu64, 0xe9b5dba58189dbbcu64,
    0x3956c25bf348b538u64, 0x59f111f1b605d019u64, 0x923f82a4af194f9bu64, 0xab1c5ed5da6d8118u64,
    0xd807aa98a3030242u64, 0x12835b0145706fbeu64, 0x243185be4ee4b28cu64, 0x550c7dc3d5ffb4e2u64,
    0x72be5d74f27b896fu64, 0x80deb1fe3b1696b1u64, 0x9bdc06a725c71235u64, 0xc19bf174cf692694u64,
    0xe49b69c19ef14ad2u64, 0xefbe4786384f25e3u64, 0x0fc19dc68b8cd5b5u64, 0x240ca1cc77ac9c65u64,
    0x2de92c6f592b0275u64, 0x4a7484aa6ea6e483u64, 0x5cb0a9dcbd41fbd4u64, 0x76f988da831153b5u64,
    0x983e5152ee66dfabu64, 0xa831c66d2db43210u64, 0xb00327c898fb213fu64, 0xbf597fc7beef0ee4u64,
    0xc6e00bf33da88fc2u64, 0xd5a79147930aa725u64, 0x06ca6351e003826fu64, 0x142929670a0e6e70u64,
    0x27b70a8546d22ffcu64, 0x2e1b21385c26c926u64, 0x4d2c6dfc5ac42aedu64, 0x53380d139d95b3dfu64,
    0x650a73548baf63deu64, 0x766a0abb3c77b2a8u64, 0x81c2c92e47edaee6u64, 0x92722c851482353bu64,
    0xa2bfe8a14cf10364u64, 0xa81a664bbc423001u64, 0xc24b8b70d0f89791u64, 0xc76c51a30654be30u64,
    0xd192e819d6ef5218u64, 0xd69906245565a910u64, 0xf40e35855771202au64, 0x106aa07032bbd1b8u64,
    0x19a4c116b8d2d0c8u64, 0x1e376c085141ab53u64, 0x2748774cdf8eeb99u64, 0x34b0bcb5e19b48a8u64,
    0x391c0cb3c5c95a63u64, 0x4ed8aa4ae3418acbu64, 0x5b9cca4f7763e373u64, 0x682e6ff3d6b2b8a3u64,
    0x748f82ee5defb2fcu64, 0x78a5636f43172f60u64, 0x84c87814a1f0ab72u64, 0x8cc702081a6439ecu64,
    0x90befffa23631e28u64, 0xa4506cebde82bde9u64, 0xbef9a3f7b2c67915u64, 0xc67178f2e372532bu64,
    0xca273eceea26619cu64, 0xd186b8c721c0c207u64, 0xeada7dd6cde0eb1eu64, 0xf57d4f7fee6ed178u64,
    0x06f067aa72176fbau64, 0x0a637dc5a2c898a6u64, 0x113f9804bef90daeu64, 0x1b710b35131c471bu64,
    0x28db77f523047d84u64, 0x32caab7b40c72493u64, 0x3c9ebe0a15c9bebcu64, 0x431d67c49c100d4cu64,
    0x4cc5d4becb3e42b6u64, 0x597f299cfc657e2au64, 0x5fcb6fab3ad6faecu64, 0x6c44198c4a475817u64
];

// A structure that represents that state of a digest computation
// for the SHA-2 256 family of digest functions
struct Engine256 {
    input_buffer: [u8, ..4],
    input_buffer_idx: uint,
    length_bytes: u64,
    H0: u32,
    H1: u32,
    H2: u32,
    H3: u32,
    H4: u32,
    H5: u32,
    H6: u32,
    H7: u32,
    W: [u32, ..64],
    W_idx: uint,
    finished: bool
}

// Convert a [u8] to a u32 in big endian format
fn to_u32(in: &[u8]) -> u32 {
    (in[0] as u32) << 24 |
    (in[1] as u32) << 16 |
    (in[2] as u32) << 8 |
    (in[3] as u32)
}

// Convert a u32 to a [u8] in big endian format
fn from_u32(in: u32, out: &mut [u8]) {
    out[0] = (in >> 24) as u8;
    out[1] = (in >> 16) as u8;
    out[2] = (in >> 8) as u8;
    out[3] = in as u8;
}

impl Engine256 {
    fn input_byte(&mut self, in: u8) {
        assert!(!self.finished)

        self.input_buffer[self.input_buffer_idx] = in;
        self.input_buffer_idx += 1;

        if (self.input_buffer_idx == 4) {
            self.input_buffer_idx = 0;
            let w = to_u32(self.input_buffer);
            self.process_word(w);
        }

        self.length_bytes += 1;
    }

    fn input_vec(&mut self, in: &[u8]) {
        assert!(!self.finished)

        let mut i = 0;

        while i < in.len() && self.input_buffer_idx != 0 {
            self.input_byte(in[i]);
            i += 1;
        }

        while in.len() - i >= 4 {
            let w = to_u32(in.slice(i, i + 4));
            self.process_word(w);
            self.length_bytes += 4;
            i += 4;
        }

        while i < in.len() {
            self.input_byte(in[i]);
            i += 1;
        }

    }

    fn reset(&mut self) {
        self.length_bytes = 0;
        self.finished = false;
        self.input_buffer_idx = 0;
        self.W_idx = 0;
    }

    fn process_word(&mut self, in: u32) {
        self.W[self.W_idx] = in;
        self.W_idx += 1;
        if (self.W_idx == 16) {
            self.W_idx = 0;
            self.process_block();
        }
    }

    fn process_block(&mut self) {
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

        for uint::range(16, 64) |t| {
            self.W[t] = sigma1(self.W[t - 2]) + self.W[t - 7] + sigma0(self.W[t - 15]) +
                self.W[t - 16];
        }

        let mut a = self.H0;
        let mut b = self.H1;
        let mut c = self.H2;
        let mut d = self.H3;
        let mut e = self.H4;
        let mut f = self.H5;
        let mut g = self.H6;
        let mut h = self.H7;

        let mut t = 0;
        for uint::range(0, 8) |_| {
            h += sum1(e) + ch(e, f, g) + K32[t] + self.W[t];
            d += h;
            h += sum0(a) + maj(a, b, c);
            t += 1;

            g += sum1(d) + ch(d, e, f) + K32[t] + self.W[t];
            c += g;
            g += sum0(h) + maj(h, a, b);
            t += 1;

            f += sum1(c) + ch(c, d, e) + K32[t] + self.W[t];
            b += f;
            f += sum0(g) + maj(g, h, a);
            t += 1;

            e += sum1(b) + ch(b, c, d) + K32[t] + self.W[t];
            a += e;
            e += sum0(f) + maj(f, g, h);
            t += 1;

            d += sum1(a) + ch(a, b, c) + K32[t] + self.W[t];
            h += d;
            d += sum0(e) + maj(e, f, g);
            t += 1;

            c += sum1(h) + ch(h, a, b) + K32[t] + self.W[t];
            g += c;
            c += sum0(d) + maj(d, e, f);
            t += 1;

            b += sum1(g) + ch(g, h, a) + K32[t] + self.W[t];
            f += b;
            b += sum0(c) + maj(c, d, e);
            t += 1;

            a += sum1(f) + ch(f, g, h) + K32[t] + self.W[t];
            e += a;
            a += sum0(b) + maj(b, c, d);
            t += 1;
        }

        self.H0 += a;
        self.H1 += b;
        self.H2 += c;
        self.H3 += d;
        self.H4 += e;
        self.H5 += f;
        self.H6 += g;
        self.H7 += h;
    }

    fn finish(&mut self) {
        if (self.finished) {
            return;
        }

        // must get length before adding padding
        let bit_length = self.length_bytes << 3;

        // add padding
        self.input_byte(128u8);

        while self.input_buffer_idx != 0 {
            self.input_byte(0u8);
        }

        // add length
        if (self.W_idx > 14) {
            for uint::range(self.W_idx, 16) |_| {
                self.process_word(0);
            }
        }

        while self.W_idx < 14 {
            self.process_word(0);
        }

        self.process_word((bit_length >> 32) as u32);
        self.process_word(bit_length as u32);

        self.finished = true;
    }

    fn result_256(&mut self, out: &mut [u8]) {
        self.finish();

        from_u32(self.H0, out.mut_slice(0, 4));
        from_u32(self.H1, out.mut_slice(4, 8));
        from_u32(self.H2, out.mut_slice(8, 12));
        from_u32(self.H3, out.mut_slice(12, 16));
        from_u32(self.H4, out.mut_slice(16, 20));
        from_u32(self.H5, out.mut_slice(20, 24));
        from_u32(self.H6, out.mut_slice(24, 28));
        from_u32(self.H7, out.mut_slice(28, 32));
    }

    fn result_224(&mut self, out: &mut [u8]) {
        self.finish();

        from_u32(self.H0, out.mut_slice(0, 4));
        from_u32(self.H1, out.mut_slice(4, 8));
        from_u32(self.H2, out.mut_slice(8, 12));
        from_u32(self.H3, out.mut_slice(12, 16));
        from_u32(self.H4, out.mut_slice(16, 20));
        from_u32(self.H5, out.mut_slice(20, 24));
        from_u32(self.H6, out.mut_slice(24, 28));
    }
}

static K32: [u32, ..64] = [
    0x428a2f98u32, 0x71374491u32, 0xb5c0fbcfu32, 0xe9b5dba5u32,
    0x3956c25bu32, 0x59f111f1u32, 0x923f82a4u32, 0xab1c5ed5u32,
    0xd807aa98u32, 0x12835b01u32, 0x243185beu32, 0x550c7dc3u32,
    0x72be5d74u32, 0x80deb1feu32, 0x9bdc06a7u32, 0xc19bf174u32,
    0xe49b69c1u32, 0xefbe4786u32, 0x0fc19dc6u32, 0x240ca1ccu32,
    0x2de92c6fu32, 0x4a7484aau32, 0x5cb0a9dcu32, 0x76f988dau32,
    0x983e5152u32, 0xa831c66du32, 0xb00327c8u32, 0xbf597fc7u32,
    0xc6e00bf3u32, 0xd5a79147u32, 0x06ca6351u32, 0x14292967u32,
    0x27b70a85u32, 0x2e1b2138u32, 0x4d2c6dfcu32, 0x53380d13u32,
    0x650a7354u32, 0x766a0abbu32, 0x81c2c92eu32, 0x92722c85u32,
    0xa2bfe8a1u32, 0xa81a664bu32, 0xc24b8b70u32, 0xc76c51a3u32,
    0xd192e819u32, 0xd6990624u32, 0xf40e3585u32, 0x106aa070u32,
    0x19a4c116u32, 0x1e376c08u32, 0x2748774cu32, 0x34b0bcb5u32,
    0x391c0cb3u32, 0x4ed8aa4au32, 0x5b9cca4fu32, 0x682e6ff3u32,
    0x748f82eeu32, 0x78a5636fu32, 0x84c87814u32, 0x8cc70208u32,
    0x90befffau32, 0xa4506cebu32, 0xbef9a3f7u32, 0xc67178f2u32
];

struct Sha512 {
    priv engine: Engine512
}

struct Sha384 {
    priv engine: Engine512
}

struct Sha512Trunc256 {
    priv engine: Engine512
}

struct Sha512Trunc224 {
    priv engine: Engine512
}

struct Sha256 {
    priv engine: Engine256
}

struct Sha224 {
    priv engine: Engine256
}

impl Sha512 {
    /**
     * Construct an new instance of a SHA-512 digest.
     */
    pub fn new() -> Sha512 {
        Sha512 {
            engine: Engine512 {
                input_buffer: [0u8, ..8],
                input_buffer_idx: 0,
                bit_counter: BitCounter { high_bit_count: 0, low_byte_count: 0 },
                H0: 0x6a09e667f3bcc908u64,
                H1: 0xbb67ae8584caa73bu64,
                H2: 0x3c6ef372fe94f82bu64,
                H3: 0xa54ff53a5f1d36f1u64,
                H4: 0x510e527fade682d1u64,
                H5: 0x9b05688c2b3e6c1fu64,
                H6: 0x1f83d9abfb41bd6bu64,
                H7: 0x5be0cd19137e2179u64,
                W: [0u64, ..80],
                W_idx: 0,
                finished: false,
            }
        }
    }
}

impl Sha384 {
    /**
     * Construct an new instance of a SHA-384 digest.
     */
    pub fn new() -> Sha384 {
        Sha384 {
            engine: Engine512 {
                input_buffer: [0u8, ..8],
                input_buffer_idx: 0,
                bit_counter: BitCounter { high_bit_count: 0, low_byte_count: 0 },
                H0: 0xcbbb9d5dc1059ed8u64,
                H1: 0x629a292a367cd507u64,
                H2: 0x9159015a3070dd17u64,
                H3: 0x152fecd8f70e5939u64,
                H4: 0x67332667ffc00b31u64,
                H5: 0x8eb44a8768581511u64,
                H6: 0xdb0c2e0d64f98fa7u64,
                H7: 0x47b5481dbefa4fa4u64,
                W: [0u64, ..80],
                W_idx: 0,
                finished: false,
            }
        }
    }
}

impl Sha512Trunc256 {
    /**
     * Construct an new instance of a SHA-512/256 digest.
     */
    pub fn new() -> Sha512Trunc256 {
        Sha512Trunc256 {
            engine: Engine512 {
                input_buffer: [0u8, ..8],
                input_buffer_idx: 0,
                bit_counter: BitCounter { high_bit_count: 0, low_byte_count: 0 },
                H0: 0x22312194fc2bf72cu64,
                H1: 0x9f555fa3c84c64c2u64,
                H2: 0x2393b86b6f53b151u64,
                H3: 0x963877195940eabdu64,
                H4: 0x96283ee2a88effe3u64,
                H5: 0xbe5e1e2553863992u64,
                H6: 0x2b0199fc2c85b8aau64,
                H7: 0x0eb72ddc81c52ca2u64,
                W: [0u64, ..80],
                W_idx: 0,
                finished: false,
            }
        }
    }
}

impl Sha512Trunc224 {
    /**
     * Construct an new instance of a SHA-512/224 digest.
     */
    pub fn new() -> Sha512Trunc224 {
        Sha512Trunc224 {
            engine: Engine512 {
                input_buffer: [0u8, ..8],
                input_buffer_idx: 0,
                bit_counter: BitCounter { high_bit_count: 0, low_byte_count: 0 },
                H0: 0x8c3d37c819544da2u64,
                H1: 0x73e1996689dcd4d6u64,
                H2: 0x1dfab7ae32ff9c82u64,
                H3: 0x679dd514582f9fcfu64,
                H4: 0x0f6d2b697bd44da8u64,
                H5: 0x77e36f7304c48942u64,
                H6: 0x3f9d85a86a1d36c8u64,
                H7: 0x1112e6ad91d692a1u64,
                W: [0u64, ..80],
                W_idx: 0,
                finished: false,
            }
        }
    }
}

impl Sha256 {
    /**
     * Construct an new instance of a SHA-256 digest.
     */
    pub fn new() -> Sha256 {
        Sha256 {
            engine: Engine256 {
                input_buffer: [0u8, ..4],
                input_buffer_idx: 0,
                length_bytes: 0,
                H0: 0x6a09e667u32,
                H1: 0xbb67ae85u32,
                H2: 0x3c6ef372u32,
                H3: 0xa54ff53au32,
                H4: 0x510e527fu32,
                H5: 0x9b05688cu32,
                H6: 0x1f83d9abu32,
                H7: 0x5be0cd19u32,
                W: [0u32, ..64],
                W_idx: 0,
                finished: false,
            }
        }
    }
}

impl Sha224 {
    /**
     * Construct an new instance of a SHA-224 digest.
     */
    pub fn new() -> Sha224 {
        Sha224 {
            engine: Engine256 {
                input_buffer: [0u8, ..4],
                input_buffer_idx: 0,
                length_bytes: 0,
                H0: 0xc1059ed8u32,
                H1: 0x367cd507u32,
                H2: 0x3070dd17u32,
                H3: 0xf70e5939u32,
                H4: 0xffc00b31u32,
                H5: 0x68581511u32,
                H6: 0x64f98fa7u32,
                H7: 0xbefa4fa4u32,
                W: [0u32, ..64],
                W_idx: 0,
                finished: false,
            }
        }
    }
}

impl Digest for Sha512 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input_vec(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.result_512(out)
    }

    fn reset(&mut self) {
        self.engine.reset();

        self.engine.H0 = 0x6a09e667f3bcc908u64;
        self.engine.H1 = 0xbb67ae8584caa73bu64;
        self.engine.H2 = 0x3c6ef372fe94f82bu64;
        self.engine.H3 = 0xa54ff53a5f1d36f1u64;
        self.engine.H4 = 0x510e527fade682d1u64;
        self.engine.H5 = 0x9b05688c2b3e6c1fu64;
        self.engine.H6 = 0x1f83d9abfb41bd6bu64;
        self.engine.H7 = 0x5be0cd19137e2179u64;
    }

    fn output_bits(&self) -> uint { 512 }
}

impl Digest for Sha384 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input_vec(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.result_384(out)
    }

    fn reset(&mut self) {
        self.engine.reset();

        self.engine.H0 = 0xcbbb9d5dc1059ed8u64;
        self.engine.H1 = 0x629a292a367cd507u64;
        self.engine.H2 = 0x9159015a3070dd17u64;
        self.engine.H3 = 0x152fecd8f70e5939u64;
        self.engine.H4 = 0x67332667ffc00b31u64;
        self.engine.H5 = 0x8eb44a8768581511u64;
        self.engine.H6 = 0xdb0c2e0d64f98fa7u64;
        self.engine.H7 = 0x47b5481dbefa4fa4u64;
    }

    fn output_bits(&self) -> uint { 384 }
}

impl Digest for Sha512Trunc256 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input_vec(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.result_256(out)
    }

    fn reset(&mut self) {
        self.engine.reset();

        self.engine.H0 = 0x22312194fc2bf72cu64;
        self.engine.H1 = 0x9f555fa3c84c64c2u64;
        self.engine.H2 = 0x2393b86b6f53b151u64;
        self.engine.H3 = 0x963877195940eabdu64;
        self.engine.H4 = 0x96283ee2a88effe3u64;
        self.engine.H5 = 0xbe5e1e2553863992u64;
        self.engine.H6 = 0x2b0199fc2c85b8aau64;
        self.engine.H7 = 0x0eb72ddc81c52ca2u64;
    }

    fn output_bits(&self) -> uint { 256 }
}

impl Digest for Sha512Trunc224 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input_vec(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.result_224(out)
    }

    fn reset(&mut self) {
        self.engine.reset();

        self.engine.H0 = 0x8c3d37c819544da2u64;
        self.engine.H1 = 0x73e1996689dcd4d6u64;
        self.engine.H2 = 0x1dfab7ae32ff9c82u64;
        self.engine.H3 = 0x679dd514582f9fcfu64;
        self.engine.H4 = 0x0f6d2b697bd44da8u64;
        self.engine.H5 = 0x77e36f7304c48942u64;
        self.engine.H6 = 0x3f9d85a86a1d36c8u64;
        self.engine.H7 = 0x1112e6ad91d692a1u64;
    }

    fn output_bits(&self) -> uint { 224 }
}

impl Digest for Sha256 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input_vec(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.result_256(out)
    }

    fn reset(&mut self) {
        self.engine.reset();

        self.engine.H0 = 0x6a09e667u32;
        self.engine.H1 = 0xbb67ae85u32;
        self.engine.H2 = 0x3c6ef372u32;
        self.engine.H3 = 0xa54ff53au32;
        self.engine.H4 = 0x510e527fu32;
        self.engine.H5 = 0x9b05688cu32;
        self.engine.H6 = 0x1f83d9abu32;
        self.engine.H7 = 0x5be0cd19u32;
    }

    fn output_bits(&self) -> uint { 256 }
}

impl Digest for Sha224 {
    fn input(&mut self, d: &[u8]) {
        self.engine.input_vec(d);
    }

    fn result(&mut self, out: &mut [u8]) {
        self.engine.result_224(out)
    }

    fn reset(&mut self) {
        self.engine.reset();

        self.engine.H0 = 0xc1059ed8u32;
        self.engine.H1 = 0x367cd507u32;
        self.engine.H2 = 0x3070dd17u32;
        self.engine.H3 = 0xf70e5939u32;
        self.engine.H4 = 0xffc00b31u32;
        self.engine.H5 = 0x68581511u32;
        self.engine.H6 = 0x64f98fa7u32;
        self.engine.H7 = 0xbefa4fa4u32;
    }

    fn output_bits(&self) -> uint { 224 }
}


#[cfg(test)]
mod tests {
    use digest::{Digest, DigestUtil};
    use sha2::{Sha512, Sha384, Sha512Trunc256, Sha512Trunc224, Sha256, Sha224};

    struct Test {
        input: ~str,
        output_str: ~str,
    }

    fn test_hash<D: Digest>(sh: &mut D, tests: &[Test]) {
        // Test that it works when accepting the message all at once
        for tests.iter().advance() |t| {
            sh.input_str(t.input);

            let out_str = sh.result_str();
            assert!(out_str == t.output_str);

            sh.reset();
        }

        // Test that it works when accepting the message in pieces
        for tests.iter().advance() |t| {
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
