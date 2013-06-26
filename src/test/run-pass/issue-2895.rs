// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::sys;

struct Cat {
    x: int
}

struct Kitty {
    x: int,
}

impl Drop for Kitty {
    fn drop(&self) {}
}

#[cfg(target_arch = "x86_64")]
pub fn main() {
    assert_eq!(sys::size_of::<Cat>(), 8 as uint);
    assert_eq!(sys::size_of::<Kitty>(), 16 as uint);
}

#[cfg(target_arch = "x86")]
#[cfg(target_arch = "arm")]
pub fn main() {
    assert_eq!(sys::size_of::<Cat>(), 4 as uint);
    assert_eq!(sys::size_of::<Kitty>(), 8 as uint);
}
