// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that qualified paths with type parameters
// fail during type checking and not during parsing

struct S;

trait Tr {
    type A;
}

impl Tr for S {
    type A = S;
}

impl S {
    fn f<T>() {}
}

type A = <S as Tr>::A::f<u8>;
//~^ ERROR type parameters are not allowed on this type
//~| NOTE type parameter not allowed
//~| ERROR ambiguous associated type
//~| NOTE ambiguous associated type
//~| NOTE specify the type using the syntax `<<S as Tr>::A as Trait>::f`

fn main() {}
