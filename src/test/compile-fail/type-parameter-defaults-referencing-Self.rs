// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a default that references `Self` which is then used in an object type.
// Issue #18956.

#![feature(default_type_params)]

trait Foo<T=Self> {
    fn method(&self);
}

fn foo(x: &Foo) { }
//~^ ERROR the type parameter `T` must be explicitly specified

fn main() { }
