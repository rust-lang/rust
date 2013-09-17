// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure the destructor is run for unit-like structs.
// xfail-fast

use std::task;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        fail!("This failure should happen.");
    }
}

fn main() {
    let x = do task::try {
        let _b = Foo;
    };
    assert_eq!(x, Err(()));
}
