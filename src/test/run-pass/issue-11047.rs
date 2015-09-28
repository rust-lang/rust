// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that static methods can be invoked on `type` aliases

#![allow(unused_variables)]

pub mod foo {
    pub mod bar {
        pub mod baz {
            pub struct Qux;

            impl Qux {
                pub fn new() {}
            }
        }
    }
}

fn main() {

    type Ham = foo::bar::baz::Qux;
    let foo = foo::bar::baz::Qux::new();  // invoke directly
    let bar = Ham::new();                 // invoke via type alias

    type StringVec = Vec<String>;
    let sv = StringVec::new();
}
