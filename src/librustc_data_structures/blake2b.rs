// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// An implementation of the Blake2b cryptographic hash function.
// The implementation closely follows: https://tools.ietf.org/html/rfc7693
//
// "BLAKE2 is a cryptographic hash function faster than MD5, SHA-1, SHA-2, and
//  SHA-3, yet is at least as secure as the latest standard SHA-3."
// according to their own website :)
//
// Indeed this implementation is two to three times as fast as our SHA-256
// implementation. If you have the luxury of being able to use crates from
// crates.io, you can go there and find still faster implementations.

use std::mem;
use std::slice;

pub struct Blake2bCtx {
    b: [u8; 128],
    h: [u64; 8],
    t: [u64; 2],
    c: usize,
    outlen: u16,
    finalized: bool
}

impl ::std::fmt::Debug for Blake2bCtx {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        try!(write!(fmt, "hash: "));
        for v in &self.h[..] {
            try!(write!(fmt, "{:x}", v));
        }
        Ok(())
    }
}

#[inline(always)]
fn b2b_g(v: &mut [u64; 16],
         a: usize,
         b: usize,
         c: usize,
         d: usize,
         x: u64,
         y: u64)
{
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

// Initialization vector
const BLAKE2B_IV: [u64; 8] = [
   0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
   0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
   0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
   0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
];

fn blake2b_compress(ctx: &mut Blake2bCtx, last: bool) {

    const SIGMA: [[usize; 16]; 12] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 ],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 ],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 ],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 ],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 ],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 ],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 ],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 ],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 ]
    ];

    let mut v: [u64; 16] = [
        ctx.h[0],
        ctx.h[1],
        ctx.h[2],
        ctx.h[3],
        ctx.h[4],
        ctx.h[5],
        ctx.h[6],
        ctx.h[7],

        BLAKE2B_IV[0],
        BLAKE2B_IV[1],
        BLAKE2B_IV[2],
        BLAKE2B_IV[3],
        BLAKE2B_IV[4],
        BLAKE2B_IV[5],
        BLAKE2B_IV[6],
        BLAKE2B_IV[7],
    ];

    v[12] ^= ctx.t[0]; // low 64 bits of offset
    v[13] ^= ctx.t[1]; // high 64 bits
    if last {
        v[14] = !v[14];
    }

    {
        // Re-interpret the input buffer in the state as an array
        // of little-endian u64s, converting them to machine
        // endianness. It's OK to modify the buffer in place
        // since this is the last time  this data will be accessed
        // before it's overwritten.

        let m: &mut [u64; 16] = unsafe {
            let b: &mut [u8; 128] = &mut ctx.b;
            ::std::mem::transmute(b)
        };

        if cfg!(target_endian = "big") {
            for word in &mut m[..] {
                *word = u64::from_le(*word);
            }
        }

        for i in 0 .. 12 {
            b2b_g(&mut v, 0, 4,  8, 12, m[SIGMA[i][ 0]], m[SIGMA[i][ 1]]);
            b2b_g(&mut v, 1, 5,  9, 13, m[SIGMA[i][ 2]], m[SIGMA[i][ 3]]);
            b2b_g(&mut v, 2, 6, 10, 14, m[SIGMA[i][ 4]], m[SIGMA[i][ 5]]);
            b2b_g(&mut v, 3, 7, 11, 15, m[SIGMA[i][ 6]], m[SIGMA[i][ 7]]);
            b2b_g(&mut v, 0, 5, 10, 15, m[SIGMA[i][ 8]], m[SIGMA[i][ 9]]);
            b2b_g(&mut v, 1, 6, 11, 12, m[SIGMA[i][10]], m[SIGMA[i][11]]);
            b2b_g(&mut v, 2, 7,  8, 13, m[SIGMA[i][12]], m[SIGMA[i][13]]);
            b2b_g(&mut v, 3, 4,  9, 14, m[SIGMA[i][14]], m[SIGMA[i][15]]);
        }
    }

    for i in 0 .. 8 {
        ctx.h[i] ^= v[i] ^ v[i + 8];
    }
}

fn blake2b_new(outlen: usize, key: &[u8]) -> Blake2bCtx {
    assert!(outlen > 0 && outlen <= 64 && key.len() <= 64);

    let mut ctx = Blake2bCtx {
        b: [0; 128],
        h: BLAKE2B_IV,
        t: [0; 2],
        c: 0,
        outlen: outlen as u16,
        finalized: false,
    };

    ctx.h[0] ^= 0x01010000 ^ ((key.len() << 8) as u64) ^ (outlen as u64);

    if key.len() > 0 {
       blake2b_update(&mut ctx, key);
       ctx.c = ctx.b.len();
    }

    ctx
}

fn blake2b_update(ctx: &mut Blake2bCtx, mut data: &[u8]) {
    assert!(!ctx.finalized, "Blake2bCtx already finalized");

    let mut bytes_to_copy = data.len();
    let mut space_in_buffer = ctx.b.len() - ctx.c;

    while bytes_to_copy > space_in_buffer {
        checked_mem_copy(data, &mut ctx.b[ctx.c .. ], space_in_buffer);

        ctx.t[0] = ctx.t[0].wrapping_add(ctx.b.len() as u64);
        if ctx.t[0] < (ctx.b.len() as u64) {
            ctx.t[1] += 1;
        }
        blake2b_compress(ctx, false);
        ctx.c = 0;

        data = &data[space_in_buffer .. ];
        bytes_to_copy -= space_in_buffer;
        space_in_buffer = ctx.b.len();
    }

    if bytes_to_copy > 0 {
        checked_mem_copy(data, &mut ctx.b[ctx.c .. ], bytes_to_copy);
        ctx.c += bytes_to_copy;
    }
}

fn blake2b_final(ctx: &mut Blake2bCtx)
{
    assert!(!ctx.finalized, "Blake2bCtx already finalized");

    ctx.t[0] = ctx.t[0].wrapping_add(ctx.c as u64);
    if ctx.t[0] < ctx.c as u64 {
        ctx.t[1] += 1;
    }

    while ctx.c < 128 {
        ctx.b[ctx.c] = 0;
        ctx.c += 1;
    }

    blake2b_compress(ctx, true);

    // Modify our buffer to little-endian format as it will be read
    // as a byte array. It's OK to modify the buffer in place since
    // this is the last time this data will be accessed.
    if cfg!(target_endian = "big") {
        for word in &mut ctx.h {
            *word = word.to_le();
        }
    }

    ctx.finalized = true;
}

#[inline(always)]
fn checked_mem_copy<T1, T2>(from: &[T1], to: &mut [T2], byte_count: usize) {
    let from_size = from.len() * mem::size_of::<T1>();
    let to_size = to.len() * mem::size_of::<T2>();
    assert!(from_size >= byte_count);
    assert!(to_size >= byte_count);
    let from_byte_ptr = from.as_ptr() as * const u8;
    let to_byte_ptr = to.as_mut_ptr() as * mut u8;
    unsafe {
        ::std::ptr::copy_nonoverlapping(from_byte_ptr, to_byte_ptr, byte_count);
    }
}

pub fn blake2b(out: &mut [u8], key: &[u8],  data: &[u8])
{
    let mut ctx = blake2b_new(out.len(), key);
    blake2b_update(&mut ctx, data);
    blake2b_final(&mut ctx);
    checked_mem_copy(&ctx.h, out, ctx.outlen as usize);
}

pub struct Blake2bHasher(Blake2bCtx);

impl ::std::hash::Hasher for Blake2bHasher {
    fn write(&mut self, bytes: &[u8]) {
        blake2b_update(&mut self.0, bytes);
    }

    fn finish(&self) -> u64 {
        assert!(self.0.outlen == 8,
                "Hasher initialized with incompatible output length");
        u64::from_le(self.0.h[0])
    }
}

impl Blake2bHasher {
    pub fn new(outlen: usize, key: &[u8]) -> Blake2bHasher {
        Blake2bHasher(blake2b_new(outlen, key))
    }

    pub fn finalize(&mut self) -> &[u8] {
        if !self.0.finalized {
            blake2b_final(&mut self.0);
        }
        debug_assert!(mem::size_of_val(&self.0.h) >= self.0.outlen as usize);
        let raw_ptr = (&self.0.h[..]).as_ptr() as * const u8;
        unsafe {
            slice::from_raw_parts(raw_ptr, self.0.outlen as usize)
        }
    }
}

impl ::std::fmt::Debug for Blake2bHasher {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(fmt, "{:?}", self.0)
    }
}

#[cfg(test)]
fn selftest_seq(out: &mut [u8], seed: u32)
{
   let mut a: u32 = 0xDEAD4BADu32.wrapping_mul(seed);
   let mut b: u32 = 1;

   for i in 0 .. out.len() {
       let t: u32 = a.wrapping_add(b);
       a = b;
       b = t;
       out[i] = ((t >> 24) & 0xFF) as u8;
   }
}

#[test]
fn blake2b_selftest()
{
    use std::hash::Hasher;

    // grand hash of hash results
    const BLAKE2B_RES: [u8; 32] = [
        0xC2, 0x3A, 0x78, 0x00, 0xD9, 0x81, 0x23, 0xBD,
        0x10, 0xF5, 0x06, 0xC6, 0x1E, 0x29, 0xDA, 0x56,
        0x03, 0xD7, 0x63, 0xB8, 0xBB, 0xAD, 0x2E, 0x73,
        0x7F, 0x5E, 0x76, 0x5A, 0x7B, 0xCC, 0xD4, 0x75
    ];

    // parameter sets
    const B2B_MD_LEN: [usize; 4] = [20, 32, 48, 64];
    const B2B_IN_LEN: [usize; 6] = [0, 3, 128, 129, 255, 1024];

    let mut data = [0u8; 1024];
    let mut md = [0u8; 64];
    let mut key = [0u8; 64];

    let mut hasher = Blake2bHasher::new(32, &[]);

    for i in 0 .. 4 {
       let outlen = B2B_MD_LEN[i];
       for j in 0 .. 6 {
            let inlen = B2B_IN_LEN[j];

            selftest_seq(&mut data[.. inlen], inlen as u32); // unkeyed hash
            blake2b(&mut md[.. outlen], &[], &data[.. inlen]);
            hasher.write(&md[.. outlen]); // hash the hash

            selftest_seq(&mut key[0 .. outlen], outlen as u32); // keyed hash
            blake2b(&mut md[.. outlen], &key[.. outlen], &data[.. inlen]);
            hasher.write(&md[.. outlen]); // hash the hash
       }
    }

    // compute and compare the hash of hashes
    let md = hasher.finalize();
    for i in 0 .. 32 {
        assert_eq!(md[i], BLAKE2B_RES[i]);
    }
}
