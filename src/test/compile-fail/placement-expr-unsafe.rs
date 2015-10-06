// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that placement in respects unsafe code checks.

#![feature(box_heap)]
#![feature(placement_in_syntax)]

fn main() {
    use std::boxed::HEAP;

    let p: *const i32 = &42;
    let _ = in HEAP { *p }; //~ ERROR requires unsafe

    let p: *const _ = &HEAP;
    let _ = in *p { 42 }; //~ ERROR requires unsafe
}
