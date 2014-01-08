// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod a {
    pub trait A {
        fn foo(&self);
    }

}
pub mod b {
    use a::A;

    pub struct B;
    impl A for B { fn foo(&self) {} }

    pub mod c {
        use b::B;

        fn foo(b: &B) {
            b.foo(); //~ ERROR: does not implement any method in scope named
        }
    }

}

fn main() {}
