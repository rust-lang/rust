// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Various scenarios in which `pub` is required in blocks

struct S;

mod m {
    fn f() {
        impl ::S {
            pub fn s(&self) {}
        }
    }
}

// ------------------------------------------------------

pub trait Tr {
    type A;
}
pub struct S1;

fn f() {
    pub struct Z;

    impl ::Tr for ::S1 {
        type A = Z; // Private-in-public error unless `struct Z` is pub
    }
}

// ------------------------------------------------------

trait Tr1 {
    type A;
    fn pull(&self) -> Self::A;
}
struct S2;

mod m1 {
    fn f() {
        pub struct Z {
            pub field: u8
        }

        impl ::Tr1 for ::S2 {
            type A = Z;
            fn pull(&self) -> Self::A { Z{field: 10} }
        }
    }
}

// ------------------------------------------------------

fn main() {
    S.s(); // Privacy error, unless `fn s` is pub
    let a = S2.pull().field; // Privacy error unless `field: u8` is pub
}
