// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test method calls with self as an argument (cross-crate)

#![allow(unknown_features)]
#![feature(box_syntax)]

// aux-build:method_self_arg2.rs
extern crate method_self_arg2;
use method_self_arg2::{Foo, Bar};

fn main() {
    let x = Foo;
    // Test external call.
    Bar::foo1(&x);
    Bar::foo2(x);
    Bar::foo3(box x);

    Bar::bar1(&x);
    Bar::bar2(x);
    Bar::bar3(box x);

    x.run_trait();

    assert!(method_self_arg2::get_count() == 2u64*2*3*3*5*5*7*7*11*11*13*13*17);
}
