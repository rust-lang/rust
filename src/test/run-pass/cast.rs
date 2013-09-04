// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// -*- rust -*-
pub fn main() {
    let i: int = 'Q' as int;
    assert_eq!(i, 0x51);
    let u: u32 = i as u32;
    assert_eq!(u, 0x51 as u32);
    assert_eq!(u, 'Q' as u32);
    assert_eq!(i as u8, 'Q' as u8);
    assert_eq!(i as u8 as i8, 'Q' as u8 as i8);
    assert_eq!(0x51u8 as char, 'Q');
    assert_eq!(true, 1 as bool);
    assert_eq!(0 as u32, false as u32);
}
