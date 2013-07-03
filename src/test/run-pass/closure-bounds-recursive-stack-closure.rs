// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensures that it's legal to create a recursive stack closure as long as
// its environment is copyable

struct R<'self> {
    // This struct is needed to create the
    // otherwise infinite type of a fn that
    // accepts itself as argument:
    c: &'self fn:Copy(&R, uint) -> uint
}

fn main() {
    // Stupid version of fibonacci.
    let fib: &fn:Copy(&R, uint) -> uint = |fib, x| {
        if x == 0 || x == 1 {
            x
        } else {
            (fib.c)(fib, x-1) + (fib.c)(fib, x-2)
        }
    };
    assert!(fib(&R { c: fib }, 7) == 13);
}
