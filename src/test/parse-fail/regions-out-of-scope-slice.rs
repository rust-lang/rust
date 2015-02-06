// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test blk region isn't supported in the front-end

fn foo(cond: bool) {
    // Here we will infer a type that uses the
    // region of the if stmt then block, but in the scope:
    let mut x; //~ ERROR foo

    if cond {
        x = &'blk [1,2,3];
    }
}

fn main() {}
