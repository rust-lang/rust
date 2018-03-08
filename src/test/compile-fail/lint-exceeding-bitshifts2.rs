// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(exceeding_bitshifts, const_err)]
#![allow(unused_variables)]
#![allow(dead_code)]

fn main() {
      let n = 1u8 << (4+3);
      let n = 1u8 << (4+4); //~ ERROR: attempt to shift left with overflow
      let n = 1i64 >> [63][0];
      let n = 1i64 >> [64][0]; // should be linting, needs to wait for const propagation

      #[cfg(target_pointer_width = "32")]
      const BITS: usize = 32;
      #[cfg(target_pointer_width = "64")]
      const BITS: usize = 64;
      let n = 1_isize << BITS; //~ ERROR: attempt to shift left with overflow
      let n = 1_usize << BITS; //~ ERROR: attempt to shift left with overflow
}
