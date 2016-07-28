// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#[repr(u8)]
enum Eu8 {
    Au8 = 23,
    Bu8 = 223,
    Cu8 = -23, //~ ERROR E0080
               //~| unary negation of unsigned integer
}

#[repr(u16)]
enum Eu16 {
    Au16 = 23,
    Bu16 = 55555,
    Cu16 = -22333, //~ ERROR E0080
                   //~| unary negation of unsigned integer
}

#[repr(u32)]
enum Eu32 {
    Au32 = 23,
    Bu32 = 3_000_000_000,
    Cu32 = -2_000_000_000, //~ ERROR E0080
                           //~| unary negation of unsigned integer
}

#[repr(u64)]
enum Eu64 {
    Au32 = 23,
    Bu32 = 3_000_000_000,
    Cu32 = -2_000_000_000, //~ ERROR E0080
                           //~| unary negation of unsigned integer
}

// u64 currently allows negative numbers, and i64 allows numbers greater than `1<<63`.  This is a
// little counterintuitive, but since the discriminant can store all the bits, and extracting it
// with a cast requires specifying the signedness, there is no loss of information in those cases.
// This also applies to isize and usize on 64-bit targets.

pub fn main() { }
