// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(more_struct_aliases)]

struct S;

trait Tr {
    fn f() {
        let s = Self {};
        //~^ ERROR expected struct, variant or union type, found Self
        let z = Self::<u8> {};
        //~^ ERROR expected struct, variant or union type, found Self
        //~| ERROR type parameters are not allowed on this type
        match s {
            Self { .. } => {}
            //~^ ERROR expected struct, variant or union type, found Self
        }
    }
}

impl Tr for S {
    fn f() {
        let s = Self {}; // OK
        let z = Self::<u8> {}; //~ ERROR type parameters are not allowed on this type
        match s {
            Self { .. } => {} // OK
        }
    }
}

impl S {
    fn g() {
        let s = Self {}; // OK
        let z = Self::<u8> {}; //~ ERROR type parameters are not allowed on this type
        match s {
            Self { .. } => {} // OK
        }
    }
}

fn main() {}
