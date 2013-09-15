// xfail-pretty

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This file is intended to test only that methods are automatically
// reachable for each numeric type, for each exported impl, with no imports
// necessary. Testing the methods of the impls is done within the source
// file for each numeric type.
pub fn main() {
// ints
    // num
    assert_eq!(15i.add(&6), 21);
    assert_eq!(15i8.add(&6i8), 21i8);
    assert_eq!(15i16.add(&6i16), 21i16);
    assert_eq!(15i32.add(&6i32), 21i32);
    assert_eq!(15i64.add(&6i64), 21i64);

// uints
    // num
    assert_eq!(15u.add(&6u), 21u);
    assert_eq!(15u8.add(&6u8), 21u8);
    assert_eq!(15u16.add(&6u16), 21u16);
    assert_eq!(15u32.add(&6u32), 21u32);
    assert_eq!(15u64.add(&6u64), 21u64);

    // times
     15u.times(|| {});

// floats
    // num
    assert_eq!(10f32.to_int().unwrap(), 10);
    assert_eq!(10f64.to_int().unwrap(), 10);
}
