// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure associated items are recommended only in appropriate contexts.

struct S {
    field: u8,
}

trait Tr {
    fn method(&self);
    type Type;
}

impl Tr for S {
    type Type = u8;

    fn method(&self) {
        let _: field;
        //~^ ERROR cannot find type `field`
        //~| NOTE not found in this scope
        let field(..);
        //~^ ERROR cannot find tuple struct/variant `field`
        //~| NOTE not found in this scope
        field;
        //~^ ERROR cannot find value `field`

        let _: Type;
        //~^ ERROR cannot find type `Type`
        let Type(..);
        //~^ ERROR cannot find tuple struct/variant `Type`
        //~| NOTE not found in this scope
        Type;
        //~^ ERROR cannot find value `Type`
        //~| NOTE not found in this scope

        let _: method;
        //~^ ERROR cannot find type `method`
        //~| NOTE not found in this scope
        let method(..);
        //~^ ERROR cannot find tuple struct/variant `method`
        //~| NOTE not found in this scope
        method;
        //~^ ERROR cannot find value `method`
    }
}

fn main() {}
