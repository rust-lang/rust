// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure speculative path resolution works properly when resolution
// adjustment happens and no extra errors is reported.

struct S {
    field: u8,
}

trait Tr {
    fn method(&self);
}

impl Tr for S {
    fn method(&self) {
        fn g() {
            // Speculative resolution of `Self` and `self` silently fails,
            // "did you mean" messages are not printed.
            field;
            //~^ ERROR unresolved value `field`
            //~| NOTE no resolution found
            method();
            //~^ ERROR unresolved function `method`
            //~| NOTE no resolution found
        }

        field;
        //~^ ERROR unresolved value `field`
        //~| NOTE did you mean `self.field`?
        method();
        //~^ ERROR unresolved function `method`
        //~| NOTE did you mean `self.method(...)`?
    }
}

fn main() {}
