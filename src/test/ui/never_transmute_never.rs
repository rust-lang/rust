// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![crate_type="lib"]

#![feature(never_type)]
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

struct Foo;

pub fn f(x: !) -> ! {
    x
}

pub fn ub() {
    // This is completely undefined behaviour,
    // but we still want to make sure it compiles.
    let x: ! = unsafe {
        std::mem::transmute::<Foo, !>(Foo)
    };
    f(x)
}
