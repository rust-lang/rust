// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(exceeding_bitshifts)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(const_indexing)]

fn main() {
      let n = 1u8 << 7;
      let n = 1u8 << 8;   //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1u16 << 15;
      let n = 1u16 << 16; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1u32 << 31;
      let n = 1u32 << 32; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1u64 << 63;
      let n = 1u64 << 64; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1i8 << 7;
      let n = 1i8 << 8;   //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1i16 << 15;
      let n = 1i16 << 16; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1i32 << 31;
      let n = 1i32 << 32; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1i64 << 63;
      let n = 1i64 << 64; //~ ERROR: bitshift exceeds the type's number of bits

      let n = 1u8 >> 7;
      let n = 1u8 >> 8;   //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1u16 >> 15;
      let n = 1u16 >> 16; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1u32 >> 31;
      let n = 1u32 >> 32; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1u64 >> 63;
      let n = 1u64 >> 64; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1i8 >> 7;
      let n = 1i8 >> 8;   //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1i16 >> 15;
      let n = 1i16 >> 16; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1i32 >> 31;
      let n = 1i32 >> 32; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1i64 >> 63;
      let n = 1i64 >> 64; //~ ERROR: bitshift exceeds the type's number of bits

      let n = 1u8;
      let n = n << 7;
      let n = n << 8; //~ ERROR: bitshift exceeds the type's number of bits

      let n = 1u8 << -8; //~ ERROR: bitshift exceeds the type's number of bits
      //~^ WARN: attempt to shift by a negative amount

      let n = 1u8 << (4+3);
      let n = 1u8 << (4+4); //~ ERROR: bitshift exceeds the type's number of bits

      #[cfg(target_pointer_width = "32")]
      const BITS: usize = 32;
      #[cfg(target_pointer_width = "64")]
      const BITS: usize = 64;

      let n = 1_isize << BITS; //~ ERROR: bitshift exceeds the type's number of bits
      let n = 1_usize << BITS; //~ ERROR: bitshift exceeds the type's number of bits


      let n = 1i8<<(1isize+-1);

      let n = 1i64 >> [63][0];
      let n = 1i64 >> [64][0]; //~ ERROR: bitshift exceeds the type's number of bits
}
