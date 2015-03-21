// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that patterns including the box syntax are gated by `box_patterns` feature gate.

fn main() {
    let x = Box::new(1);

    match x {
        box 1 => (),
        //~^ box pattern syntax is experimental
        //~| add #![feature(box_patterns)] to the crate attributes to enable
        _     => ()
    };
}
