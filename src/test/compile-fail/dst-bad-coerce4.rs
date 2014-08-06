// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Attempt to coerce from unsized to sized.

struct Fat<Sized? T> {
    ptr: T
}

pub fn main() {
    // With a vec of ints.
    let f1: &Fat<[int]> = &Fat { ptr: [1, 2, 3] };
    let f2: &Fat<[int, ..3]> = f1;
    //~^ ERROR mismatched types: expected `&Fat<[int, .. 3]>`, found `&Fat<[int]>`
}
