// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]

fn foo<F>(f: F)
    where F: FnOnce()
{
}

fn main() {
    // Test that this closure is inferred to `FnOnce`
    // because it moves from `y.as<Option::Some>.0`:
    let x = Some(vec![1, 2, 3]);
    foo(|| {
        match x {
            Some(y) => { }
            None => { }
        }
    });

    // Test that this closure is inferred to `FnOnce`
    // because it moves from `y.0`:
    let y = (vec![1, 2, 3], 0);
    foo(|| {
        let x = y.0;
    });
}
