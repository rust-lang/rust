// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we detect unused values that are cast to other things.
// The problem was specified to casting to `*`, as creating unsafe
// pointers was not being fully checked. Issue #20791.

fn main() {
    let x: &i32;
    let y = x as *const i32; //~ ERROR use of possibly uninitialized variable: `*x`
}
