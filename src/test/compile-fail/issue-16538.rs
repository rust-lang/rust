// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod Y {
    pub type X = usize;
    extern {
        pub static x: *const usize;
    }
    pub fn foo(value: *const X) -> *const X {
        value
    }
}

static foo: *const Y::X = Y::foo(Y::x as *const Y::X);
//~^ ERROR the trait `core::marker::Sync` is not implemented for the type
//~| ERROR cannot refer to other statics by value, use the address-of operator or a constant instead
//~| ERROR E0015

fn main() {}
