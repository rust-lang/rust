// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod m2 {
    pub enum Foo {
        A,
        B(isize),
        C { a: isize },
    }

    impl Foo {
        pub fn foo() {}
        pub fn bar(&self) {}
    }
}

mod m {
    pub use m2::Foo::*;
}

pub fn main() {
    use m2::Foo::*;

    foo(); //~ ERROR cannot find function `foo` in this scope
    m::foo(); //~ ERROR cannot find function `foo` in module `m`
    bar(); //~ ERROR cannot find function `bar` in this scope
    m::bar(); //~ ERROR cannot find function `bar` in module `m`
}
