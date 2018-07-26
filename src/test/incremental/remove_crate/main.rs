// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that removing an upstream crate does not cause any trouble.

// revisions:rpass1 rpass2
// aux-build:extern_crate.rs

#[cfg(rpass1)]
extern crate extern_crate;

pub fn main() {
    #[cfg(rpass1)]
    {
        extern_crate::foo(1);
    }

    #[cfg(rpass2)]
    {
        foo(1);
    }
}

#[cfg(rpass2)]
pub fn foo(_: u8) {

}
