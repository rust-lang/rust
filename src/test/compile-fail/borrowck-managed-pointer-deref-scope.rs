// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verify that managed pointers scope is treated like ownoed pointers.
// regresion test for #11586

#![feature(managed_boxes)]

use std::gc::{GC, Gc};

fn foo(x: &Gc<int>) -> &int {
    match x {
        &ref y => {
            &**y // Do not expect an error here
        }
    }
}

fn bar() {
    let a = 3i;
    let mut y = &a;
    if true {
        let x = box(GC) 3i;
        y = &*x; //~ ERROR `*x` does not live long enough
    }
}

fn main() {}
