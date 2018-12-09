// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn dec_read_dec(i: &mut i32) -> i32 {
    *i -= 1;
    let ret = *i;
    *i -= 1;
    ret
}

#[allow(clippy::trivially_copy_pass_by_ref)]
pub fn minus_1(i: &i32) -> i32 {
    dec_read_dec(&mut i.clone())
}

fn main() {
    let mut i = 10;
    assert_eq!(minus_1(&i), 9);
    assert_eq!(i, 10);
    assert_eq!(dec_read_dec(&mut i), 9);
    assert_eq!(i, 8);
}
