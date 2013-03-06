// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    fail_unless!(0xffffffffu32 == (-1 as u32));
    fail_unless!(4294967295u32 == (-1 as u32));
    fail_unless!(0xffffffffffffffffu64 == (-1 as u64));
    fail_unless!(18446744073709551615u64 == (-1 as u64));

    fail_unless!(-2147483648i32 - 1i32 == 2147483647i32);
    fail_unless!(-9223372036854775808i64 - 1i64 == 9223372036854775807i64);
}
