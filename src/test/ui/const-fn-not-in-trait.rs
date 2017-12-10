// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that const fn is illegal in a trait declaration, whether or
// not a default is provided.

#![feature(const_fn)]

trait Foo {
    const fn f() -> u32;
    //~^ ERROR trait fns cannot be declared const
    //~| NOTE trait fns cannot be const
    const fn g() -> u32 { 0 }
    //~^ ERROR trait fns cannot be declared const
    //~| NOTE trait fns cannot be const
}

fn main() { }
