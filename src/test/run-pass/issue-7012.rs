// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
# Comparison of static arrays

The expected behaviour would be that test==test1, therefore 'true'
would be printed, however the below prints false.
*/

struct signature<'a> { pattern : &'a [u32] }

static test1: signature<'static> =  signature {
  pattern: &[0x243f6a88u32,0x85a308d3u32,0x13198a2eu32,0x03707344u32,0xa4093822u32,0x299f31d0u32]
};

pub fn main() {
  let test = &[0x243f6a88u32,0x85a308d3u32,0x13198a2eu32,0x03707344u32,0xa4093822u32,0x299f31d0u32];
  println!("{}",test==test1.pattern);
}
