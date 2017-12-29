// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that safe fns are not a subtype of unsafe fns.

trait Foo {
    unsafe fn len(&self) -> u32;
}

impl Foo for u32 {
    fn len(&self) -> u32 { *self }
    //~^ ERROR method `len` has an incompatible type for trait
    //~| expected type `unsafe fn(&u32) -> u32`
    //~| found type `fn(&u32) -> u32`
}

fn main() { }
