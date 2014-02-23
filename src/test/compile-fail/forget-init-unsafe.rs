// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::intrinsics::{init, forget};

// Test that the `forget` and `init` intrinsics are really unsafe
pub fn main() {
    let stuff = init::<int>(); //~ ERROR call to unsafe function requires unsafe
    forget(stuff);             //~ ERROR call to unsafe function requires unsafe
}
