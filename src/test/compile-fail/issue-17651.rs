// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that moves of unsized values within closures are caught
// and rejected.

#![feature(box_syntax)]

fn main() {
    (|| box *[0_usize].as_slice())();
    //~^ ERROR cannot move out of borrowed content
    //~^^ ERROR cannot move a value of type [usize]
}
