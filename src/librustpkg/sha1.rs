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
 *
 * This implementation has not been reviewed for cryptographic uses.
 * As such, all cryptographic uses of this implementation are strongly
 * discouraged.
 */

use std::num::Zero;
use std::vec;
use std::vec::bytes::{MutableByteVector, copy_memory};
use extra::hex::ToHex;

/// Write a u32 into a vector, which must be 4 bytes long. The value is written in big-endian
/// format.
fn write_u32_be(dst: &mut[u8], input: u32) {
    use std::cast::transmute;
    use std::unstable::intrinsics::to_be32;
    assert!(dst.len() == 4);
    unsafe {
        let x: *mut i32 = transmute(dst.unsafe_mut_ref(0));
        *x = to_be32(input as i32);
    }
}

/// Read a vector of bytes into a vector of u32s. The values are read in big-endian format.
fn read_u32v_be(dst: &mut[u32], input: &[u8]) {
    use std::cast::transmute;
    use std::unstable::intrinsics::to_be32;
    assert!(dst.len() * 4 == input.len());
    unsafe {
        let mut x: *mut i32 = transmute(dst.unsafe_mut_ref(0));
        let mut y: *i32 = transmute(input.unsafe_ref(0));
        for _ in range(0, dst.len()) {
            *x = to_be32(*y);
            x = x.offset(1);
            y = y.offset(1);
        }
    }
}

trait ToBits {
    /// Convert the value in bytes to the number of bits, a tuple where the 1st item is the
    /// high-order value and the 2nd item is the low order value.
    fn to_bits(self) -> (Self, Self);
}

impl ToBits for u64 {
    fn to_bits(self) -> (u64, u64) {
        return (self >> 61, self << 3);
    }
}

/// Adds the specified number of bytes to the bit count. fail!() if this would cause numeric
/// overflow.
fn add_bytes_to_bits<T: Int + CheckedAdd + ToBits>(bits: T, bytes: T) -> T {
    let (new_high_bits, new_low_bits) = bytes.to_bits();

    if new_high_bits > Zero::zero() {
        fail!("Numeric overflow occured.")
    }

    match bits.checked_add(&new_low_bits) {
        Some(x) => return x,
        None => fail!("Numeric overflow occured.")
    }
}

/// A FixedBuffer, likes its name implies, is a fixed size buffer. When the buffer becomes full, it
/// must be processed. The input() method takes care of processing and then clearing the buffer
/// automatically. However, other methods do not and require the caller to process the buffer. Any
/// method that modifies the buffer directory or provides the caller with bytes that can be modifies
/// results in those bytes being marked as used by the buffer.
trait FixedBuffer {
    /// Input a vector of bytes. If the buffer becomes full, process it with the provided
    /// function and then clear the buffer.
    fn input(&mut self, input: &[u8], func: |&[u8]|);

    /// Reset the buffer.
    fn reset(&mut self);

    /// Zero the buffer up until the specified index. The buffer position currently must not be
    /// greater than that index.
    fn zero_until(&mut self, idx: uint);

    /// Get a slice of the buffer of the specified size. There must be at least that many bytes
    /// remaining in the buffer.
    fn next<'s>(&'s mut self, len: uint) -> &'s mut [u8];

    /// Get the current buffer. The buffer must already be full. This clears the buffer as well.
    fn full_buffer<'s>(&'s mut self) -> &'s [u8];

    /// Get the current position of the buffer.
    fn position(&self) -> uint;

    /// Get the number of bytes remaining in the buffer until it is full.
    fn remaining(&self) -> uint;

    /// Get the size of the buffer
    fn size(&self) -> uint;
}

/// A fixed size buffer of 64 bytes useful for cryptographic operations.
struct FixedBuffer64 {
    priv buffer: [u8, ..64],
    priv buffer_idx: uint,
}

impl FixedBuffer64 {
    /// Create a new buffer
    fn new() -> FixedBuffer64 {
        return FixedBuffer64 {
            buffer: [0u8, ..64],
            buffer_idx: 0
        };
    }
}

impl FixedBuffer for FixedBuffer64 {
    fn input(&mut self, input: &[u8], func: |&[u8]|) {
        let mut i = 0;

        let size = 64;

        // If there is already data in the buffer, copy as much as we can into it and process
        // the data if the buffer becomes full.
        if self.buffer_idx != 0 {
            let buffer_remaining = size - self.buffer_idx;
            if input.len() >= buffer_remaining {
                    copy_memory(
                        self.buffer.mut_slice(self.buffer_idx, size),
                        input.slice_to(buffer_remaining));
                self.buffer_idx = 0;
                func(self.buffer);
                i += buffer_remaining;
            } else {
                copy_memory(
                    self.buffer.mut_slice(self.buffer_idx, self.buffer_idx + input.len()),
                    input);
                self.buffer_idx += input.len();
                return;
            }
        }

        // While we have at least a full buffer size chunks's worth of data, process that data
        // without copying it into the buffer
        while input.len() - i >= size {
            func(input.slice(i, i + size));
            i += size;
        }

        // Copy any input data into the buffer. At this point in the method, the ammount of
        // data left in the input vector will be less than the buffer size and the buffer will
        // be empty.
        let input_remaining = input.len() - i;
        copy_memory(
            self.buffer.mut_slice(0, input_remaining),
            input.slice_from(i));
        self.buffer_idx += input_remaining;
    }

    fn reset(&mut self) {
        self.buffer_idx = 0;
    }

    fn zero_until(&mut self, idx: uint) {
        assert!(idx >= self.buffer_idx);
        self.buffer.mut_slice(self.buffer_idx, idx).set_memory(0);
        self.buffer_idx = idx;
    }

    fn next<'s>(&'s mut self, len: uint) -> &'s mut [u8] {
        self.buffer_idx += len;
        return self.buffer.mut_slice(self.buffer_idx - len, self.buffer_idx);
    }

    fn full_buffer<'s>(&'s mut self) -> &'s [u8] {
        assert!(self.buffer_idx == 64);
        self.buffer_idx = 0;
        return self.buffer.slice_to(64);
    }

    fn position(&self) -> uint { self.buffer_idx }

    fn remaining(&self) -> uint { 64 - self.buffer_idx }

    fn size(&self) -> uint { 64 }
}

/// The StandardPadding trait adds a method useful for various hash algorithms to a FixedBuffer
/// struct.
trait StandardPadding {
    /// Add standard padding to the buffer. The buffer must not be full when this method is called
    /// and is guaranteed to have exactly rem remaining bytes when it returns. If there are not at
    /// least rem bytes available, the buffer will be zero padded, processed, cleared, and then
    /// filled with zeros again until only rem bytes are remaining.
    fn standard_padding(&mut self, rem: uint, func: |&[u8]|);
}

impl <T: FixedBuffer> StandardPadding for T {
    fn standard_padding(&mut self, rem: uint, func: |&[u8]|) {
        let size = self.size();

        self.next(1)[0] = 128;

        if self.remaining() < rem {
            self.zero_until(size);
            func(self.full_buffer());
        }

        self.zero_until(size - rem);
    }
}

/**
 * The Digest trait specifies an interface common to digest functions, such as SHA-1 and the SHA-2
 * family of digest functions.
 */
pub trait Digest {
    /**
     * Provide message data.
     *
     * # Arguments
     *
     * * input - A vector of message data
     */
    fn input(&mut self, input: &[u8]);

    /**
     * Retrieve the digest result. This method may be called multiple times.
     *
     * # Arguments
     *
     * * out - the vector to hold the result. Must be large enough to contain output_bits().
     */
    fn result(&mut self, out: &mut [u8]);

    /**
     * Reset the digest. This method must be called after result() and before supplying more
     * data.
     */
    fn reset(&mut self);

    /**
     * Get the output size in bits.
     */
    fn output_bits(&self) -> uint;

    /**
     * Convenience function that feeds a string into a digest.
     *
     * # Arguments
     *
     * * `input` The string to feed into the digest
     */
    fn input_str(&mut self, input: &str) {
        self.input(input.as_bytes());
    }

    /**
     * Convenience function that retrieves the result of a digest as a
     * newly allocated vec of bytes.
     */
    fn result_bytes(&mut self) -> ~[u8] {
        let mut buf = vec::from_elem((self.output_bits()+7)/8, 0u8);
        self.result(buf);
        buf
    }

    /**
     * Convenience function that retrieves the result of a digest as a
     * ~str in hexadecimal format.
     */
    fn result_str(&mut self) -> ~str {
        self.result_bytes().to_hex()
    }
}

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
    st.length_bits = add_bytes_to_bits(st.length_bits, msg.len() as u64);
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
    fn reset(&mut self) {
        self.length_bits = 0;
        self.h[0] = 0x67452301u32;
        self.h[1] = 0xEFCDAB89u32;
        self.h[2] = 0x98BADCFEu32;
        self.h[3] = 0x10325476u32;
        self.h[4] = 0xC3D2E1F0u32;
        self.buffer.reset();
        self.computed = false;
    }
    fn input(&mut self, msg: &[u8]) { add_input(self, msg); }
    fn result(&mut self, out: &mut [u8]) { return mk_result(self, out); }
    fn output_bits(&self) -> uint { 160 }
}

#[cfg(test)]
mod tests {
    use std::num::Bounded;
    use std::rand::{IsaacRng, Rng};
    use std::vec;
    use extra::hex::FromHex;
    use super::{Digest, Sha1, add_bytes_to_bits};

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

    /// Feed 1,000,000 'a's into the digest with varying input sizes and check that the result is
    /// correct.
    fn test_digest_1million_random<D: Digest>(digest: &mut D, blocksize: uint, expected: &str) {
        let total_size = 1000000;
        let buffer = vec::from_elem(blocksize * 2, 'a' as u8);
        let mut rng = IsaacRng::new_unseeded();
        let mut count = 0;

        digest.reset();

        while count < total_size {
            let next: uint = rng.gen_range(0, 2 * blocksize + 1);
            let remaining = total_size - count;
            let size = if next > remaining { remaining } else { next };
            digest.input(buffer.slice_to(size));
            count += size;
        }

        let result_str = digest.result_str();
        let result_bytes = digest.result_bytes();

        assert_eq!(expected, result_str.as_slice());
        assert_eq!(expected.from_hex().unwrap(), result_bytes);
    }

    #[test]
    fn test_1million_random_sha1() {
        let mut sh = Sha1::new();
        test_digest_1million_random(
            &mut sh,
            64,
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    }

    // A normal addition - no overflow occurs
    #[test]
    fn test_add_bytes_to_bits_ok() {
        assert!(add_bytes_to_bits::<u64>(100, 10) == 180);
    }

    // A simple failure case - adding 1 to the max value
    #[test]
    #[should_fail]
    fn test_add_bytes_to_bits_overflow() {
        add_bytes_to_bits::<u64>(Bounded::max_value(), 1);
    }
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use super::Sha1;

    #[bench]
    pub fn sha1_10(bh: & mut BenchHarness) {
        let mut sh = Sha1::new();
        let bytes = [1u8, ..10];
        bh.iter(|| sh.input(bytes));
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn sha1_1k(bh: & mut BenchHarness) {
        let mut sh = Sha1::new();
        let bytes = [1u8, ..1024];
        bh.iter(|| sh.input(bytes));
        bh.bytes = bytes.len() as u64;
    }

    #[bench]
    pub fn sha1_64k(bh: & mut BenchHarness) {
        let mut sh = Sha1::new();
        let bytes = [1u8, ..65536];
        bh.iter(|| sh.input(bytes));
        bh.bytes = bytes.len() as u64;
    }
}
