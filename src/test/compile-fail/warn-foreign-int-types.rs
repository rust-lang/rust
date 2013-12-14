// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[forbid(ctypes)];
#[allow(dead_code)];

mod xx {
    extern {
        pub fn strlen(str: *u8) -> uint; //~ ERROR found rust type `uint`
        pub fn foo(x: int, y: uint); //~ ERROR found rust type `int`
        //~^ ERROR found rust type `uint`
    }
}

fn main() {
}
