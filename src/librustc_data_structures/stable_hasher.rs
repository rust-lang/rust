// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hash::Hasher;
use std::marker::PhantomData;
use std::mem;
use blake2b::Blake2bHasher;
use rustc_serialize::leb128;
use rustc_i128::{u128,i128};

fn write_unsigned_leb128_to_buf(buf: &mut [u8; 16], value: u64) -> usize {
    leb128::write_unsigned_leb128_to(value as u128, |i, v| buf[i] = v)
}

fn write_signed_leb128_to_buf(buf: &mut [u8; 16], value: i64) -> usize {
    leb128::write_signed_leb128_to(value as i128, |i, v| buf[i] = v)
}

/// When hashing something that ends up affecting properties like symbol names. We
/// want these symbol names to be calculated independent of other factors like
/// what architecture you're compiling *from*.
///
/// The hashing just uses the standard `Hash` trait, but the implementations of
/// `Hash` for the `usize` and `isize` types are *not* architecture independent
/// (e.g. they has 4 or 8 bytes). As a result we want to avoid `usize` and
/// `isize` completely when hashing.
///
/// To do that, we encode all integers to be hashed with some
/// arch-independent encoding.
///
/// At the moment, we pass i8/u8 straight through and encode
/// all other integers using leb128.
///
/// This hasher currently always uses the stable Blake2b algorithm
/// and allows for variable output lengths through its type
/// parameter.
#[derive(Debug)]
pub struct StableHasher<W> {
    state: Blake2bHasher,
    bytes_hashed: u64,
    width: PhantomData<W>,
}

pub trait StableHasherResult: Sized {
    fn finish(hasher: StableHasher<Self>) -> Self;
}

impl<W: StableHasherResult> StableHasher<W> {
    pub fn new() -> Self {
        StableHasher {
            state: Blake2bHasher::new(mem::size_of::<W>(), &[]),
            bytes_hashed: 0,
            width: PhantomData,
        }
    }

    pub fn finish(self) -> W {
        W::finish(self)
    }
}

impl StableHasherResult for [u8; 20] {
    fn finish(mut hasher: StableHasher<Self>) -> Self {
        let mut result: [u8; 20] = [0; 20];
        result.copy_from_slice(hasher.state.finalize());
        result
    }
}

impl StableHasherResult for u64 {
    fn finish(mut hasher: StableHasher<Self>) -> Self {
        hasher.state.finalize();
        hasher.state.finish()
    }
}

impl<W> StableHasher<W> {
    #[inline]
    pub fn finalize(&mut self) -> &[u8] {
        self.state.finalize()
    }

    #[inline]
    pub fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }

    #[inline]
    fn write_uleb128(&mut self, value: u64) {
        let mut buf = [0; 16];
        let len = write_unsigned_leb128_to_buf(&mut buf, value);
        self.state.write(&buf[..len]);
        self.bytes_hashed += len as u64;
    }

    #[inline]
    fn write_ileb128(&mut self, value: i64) {
        let mut buf = [0; 16];
        let len = write_signed_leb128_to_buf(&mut buf, value);
        self.state.write(&buf[..len]);
        self.bytes_hashed += len as u64;
    }
}

// For the non-u8 integer cases we leb128 encode them first. Because small
// integers dominate, this significantly and cheaply reduces the number of
// bytes hashed, which is good because blake2b is expensive.
impl<W> Hasher for StableHasher<W> {
    fn finish(&self) -> u64 {
        panic!("use StableHasher::finish instead");
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
        self.bytes_hashed += bytes.len() as u64;
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.state.write_u8(i);
        self.bytes_hashed += 1;
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.write_uleb128(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_uleb128(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.write_uleb128(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_uleb128(i as u64);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.state.write_i8(i);
        self.bytes_hashed += 1;
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.write_ileb128(i as i64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.write_ileb128(i as i64);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_ileb128(i);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.write_ileb128(i as i64);
    }
}
