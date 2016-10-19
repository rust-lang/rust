// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deny(const_err)]

fn main() {
    #[cfg(target_pointer_width = "32")]
    const I: isize = -2147483648isize;
    #[cfg(target_pointer_width = "64")]
    const I: isize = -9223372036854775808isize;
    assert_eq!(::std::i32::MIN as u64, 0xffffffff80000000);
    assert_eq!(-2147483648isize as u64, 0xffffffff80000000);
    assert_eq!(::std::i64::MIN as u64, 0x8000000000000000);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(-9223372036854775808isize as u64, 0x8000000000000000);
    #[cfg(target_pointer_width = "32")]
    assert_eq!(-9223372036854775808isize as u64, 0);
    const J: usize = ::std::i32::MAX as usize;
    const K: usize = -1i32 as u32 as usize;
    const L: usize = ::std::i32::MIN as usize;
    const M: usize = ::std::i64::MIN as usize;
    match 5 {
        J => {},
        K => {},
        L => {},
        M => {},
        _ => {}
    }
    match 5 {
        I => {},
        _ => {}
    }
}
