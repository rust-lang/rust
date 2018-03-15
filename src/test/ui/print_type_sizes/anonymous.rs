// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z print-type-sizes

// All of the types that occur in this function are uninteresting, in
// that one cannot control the sizes of these types with the same sort
// of enum-variant manipulation tricks.

pub fn main() {
    let _byte: u8 = 0;
    let _word: usize = 0;
    let _tuple: (u8, usize)= (0, 0);
    let _array: [u8; 128] = [0; 128];
    let _fn: fn (u8) -> u8 = id;
    let _diverging: fn (u8) -> ! = bye;

    fn id(x: u8) -> u8 { x };
    fn bye(_: u8) -> ! { loop { } }
}
