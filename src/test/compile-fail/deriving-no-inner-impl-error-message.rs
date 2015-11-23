// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct NoCloneOrEq;

#[derive(PartialEq)]
struct E {
    x: NoCloneOrEq //~ ERROR binary operation `==` cannot be applied to type `NoCloneOrEq`
         //~^ ERROR binary operation `!=` cannot be applied to type `NoCloneOrEq`
}
#[derive(Clone)]
struct C {
    x: NoCloneOrEq //~ ERROR the trait `core::clone::Clone` is not implemented for the type
        //~^ ERROR the trait `core::clone::Clone` is not implemented for the type `NoCloneOrEq`
}


fn main() {}
