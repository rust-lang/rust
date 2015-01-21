// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that slicing syntax gives errors if we have not implemented the trait.

struct Foo;

fn main() {
    let x = Foo;
    &x[]; //~ ERROR cannot index a value of type `Foo`
    &x[Foo..]; //~ ERROR cannot index a value of type `Foo`
    &x[..Foo]; //~ ERROR cannot index a value of type `Foo`
    &x[Foo..Foo]; //~ ERROR cannot index a value of type `Foo`
}
