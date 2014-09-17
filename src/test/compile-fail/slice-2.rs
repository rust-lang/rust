// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that slicing syntax gives errors if we have not implemented the trait.

struct Foo;

fn main() {
    let x = Foo;
    x[]; //~ ERROR cannot take a slice of a value with type `Foo`
    x[Foo..]; //~ ERROR cannot take a slice of a value with type `Foo`
    x[..Foo]; //~ ERROR cannot take a slice of a value with type `Foo`
    x[Foo..Foo]; //~ ERROR cannot take a slice of a value with type `Foo`
    x[mut]; //~ ERROR cannot take a mutable slice of a value with type `Foo`
    x[mut Foo..]; //~ ERROR cannot take a mutable slice of a value with type `Foo`
    x[mut ..Foo]; //~ ERROR cannot take a mutable slice of a value with type `Foo`
    x[mut Foo..Foo]; //~ ERROR cannot take a mutable slice of a value with type `Foo`
}
