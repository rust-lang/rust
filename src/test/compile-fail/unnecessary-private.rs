// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    pub struct A; //~ ERROR: visibility has no effect
    pub enum B {} //~ ERROR: visibility has no effect
    pub trait C { //~ ERROR: visibility has no effect
        pub fn foo() {} //~ ERROR: visibility has no effect
    }
    impl A {
        pub fn foo() {} //~ ERROR: visibility has no effect
    }

    struct D {
        pub foo: int, //~ ERROR: visibility has no effect
    }
    pub fn foo() {} //~ ERROR: visibility has no effect
    pub mod bar {} //~ ERROR: visibility has no effect
}
