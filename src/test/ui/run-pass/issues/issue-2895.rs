// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::mem;

struct Cat {
    x: isize
}

struct Kitty {
    x: isize,
}

impl Drop for Kitty {
    fn drop(&mut self) {}
}

#[cfg(target_pointer_width = "64")]
pub fn main() {
    assert_eq!(mem::size_of::<Cat>(), 8 as usize);
    assert_eq!(mem::size_of::<Kitty>(), 8 as usize);
}

#[cfg(target_pointer_width = "32")]
pub fn main() {
    assert_eq!(mem::size_of::<Cat>(), 4 as usize);
    assert_eq!(mem::size_of::<Kitty>(), 4 as usize);
}
