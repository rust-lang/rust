// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-pretty-expanded

// This file is intended to test only that methods are automatically
// reachable for each numeric type, for each exported impl, with no imports
// necessary. Testing the methods of the impls is done within the source
// file for each numeric type.

use std::ops::Add;
use std::num::ToPrimitive;

pub fn main() {
// ints
    // num
    assert_eq!(15_isize.add(6_isize), 21_isize);
    assert_eq!(15_i8.add(6i8), 21_i8);
    assert_eq!(15_i16.add(6i16), 21_i16);
    assert_eq!(15_i32.add(6i32), 21_i32);
    assert_eq!(15_i64.add(6i64), 21_i64);

// uints
    // num
    assert_eq!(15_usize.add(6_usize), 21_usize);
    assert_eq!(15_u8.add(6u8), 21_u8);
    assert_eq!(15_u16.add(6u16), 21_u16);
    assert_eq!(15_u32.add(6u32), 21_u32);
    assert_eq!(15_u64.add(6u64), 21_u64);

// floats
    // num
    assert_eq!(10_f32.to_i32().unwrap(), 10);
    assert_eq!(10_f64.to_i32().unwrap(), 10);
}
