// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Here we are checking that a reasonable error msg is provided.
//
// The current message is not ideal, but we used to say "borrowed
// pointer has lifetime &, but the borrowed value only has lifetime &"
// which is definitely no good.


fn get() -> &int {
    //~^ NOTE borrowed pointer must be valid for the anonymous lifetime #1 defined on
    //~^^ NOTE ...but borrowed value is only valid for the block at
    let x = 3;
    return &x;
    //~^ ERROR illegal borrow
}

fn main() {}
