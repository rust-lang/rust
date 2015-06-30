// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_type_parameters)]

pub fn f<T>() {} //~ ERROR unused type parameter

pub struct S;
impl S {
    pub fn m<T>() {} //~ ERROR unused type parameter
}

trait Trait {
    fn m<T>() {} //~ ERROR unused type parameter
}

// Now test allow attributes

pub mod allow {
    #[allow(unused_type_parameters)]
    pub fn f<T>() {}

    pub struct S;
    impl S {
        #[allow(unused_type_parameters)]
        pub fn m<T>() {}
    }

    trait Trait {
        #[allow(unused_type_parameters)]
        fn m<T>() {}
    }
}

fn main() {}
