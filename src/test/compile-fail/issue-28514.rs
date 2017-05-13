// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(private_in_public)]

pub use inner::C;

mod inner {
    trait A {
        fn a(&self) { }
    }

    pub trait B {
        fn b(&self) { }
    }

    pub trait C: A + B { //~ ERROR private trait `inner::A` in public interface
                         //~^ WARN will become a hard error
        fn c(&self) { }
    }

    impl A for i32 {}
    impl B for i32 {}
    impl C for i32 {}

}

fn main() {
    // A is private
    // B is pub, not reexported
    // C : A + B is pub, reexported

    // 0.a(); // can't call
    // 0.b(); // can't call
    0.c(); // ok

    C::a(&0); // can call
    C::b(&0); // can call
    C::c(&0); // ok
}
