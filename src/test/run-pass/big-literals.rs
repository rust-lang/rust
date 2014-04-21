// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    assert_eq!(0xffffffffu32, (-1 as u32));
    assert_eq!(4294967295u32, (-1 as u32));
    assert_eq!(0xffffffffffffffffu64, (-1 as u64));
    assert_eq!(18446744073709551615u64, (-1 as u64));

    assert_eq!(-2147483648i32 - 1i32, 2147483647i32);
    assert_eq!(-9223372036854775808i64 - 1i64, 9223372036854775807i64);
    assert_eq!(-9223372036854775808i - 1, 9223372036854775807);
}
