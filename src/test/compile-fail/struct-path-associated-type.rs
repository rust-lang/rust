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
    type A;
}

impl Tr for S {
    type A = S;
}

fn f<T: Tr>() {
    let s = T::A {};
    //~^ ERROR expected struct, variant or union type, found associated type
    let z = T::A::<u8> {};
    //~^ ERROR expected struct, variant or union type, found associated type
    //~| ERROR type parameters are not allowed on this type
    match S {
        T::A {} => {}
        //~^ ERROR expected struct, variant or union type, found associated type
    }
}

fn g<T: Tr<A = S>>() {
    let s = T::A {}; // OK
    let z = T::A::<u8> {}; //~ ERROR type parameters are not allowed on this type
    match S {
        T::A {} => {} // OK
    }
}

fn main() {
    let s = S::A {}; //~ ERROR ambiguous associated type
    let z = S::A::<u8> {}; //~ ERROR ambiguous associated type
    //~^ ERROR type parameters are not allowed on this type
    match S {
        S::A {} => {} //~ ERROR ambiguous associated type
    }
}
