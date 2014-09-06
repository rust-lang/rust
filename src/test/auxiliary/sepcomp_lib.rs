// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C codegen-units=3 --crate-type=rlib,dylib

pub mod a {
    pub fn one() -> uint {
        1
    }
}

pub mod b {
    pub fn two() -> uint {
        2
    }
}

pub mod c {
    use a::one;
    use b::two;
    pub fn three() -> uint {
        one() + two()
    }
}
