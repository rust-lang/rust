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
        //~^ ERROR unresolved type `field`
        //~| NOTE no resolution found
        let field(..);
        //~^ ERROR unresolved tuple struct/variant `field`
        //~| NOTE no resolution found
        field;
        //~^ ERROR unresolved value `field`
        //~| NOTE did you mean `self.field`?

        let _: Type;
        //~^ ERROR unresolved type `Type`
        //~| NOTE did you mean `Self::Type`?
        let Type(..);
        //~^ ERROR unresolved tuple struct/variant `Type`
        //~| NOTE no resolution found
        Type;
        //~^ ERROR unresolved value `Type`
        //~| NOTE no resolution found

        let _: method;
        //~^ ERROR unresolved type `method`
        //~| NOTE no resolution found
        let method(..);
        //~^ ERROR unresolved tuple struct/variant `method`
        //~| NOTE no resolution found
        method;
        //~^ ERROR unresolved value `method`
        //~| NOTE did you mean `self.method(...)`?
    }
}

fn main() {}
