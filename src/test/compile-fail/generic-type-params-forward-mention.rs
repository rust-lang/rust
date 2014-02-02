// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(default_type_params)];

// Ensure that we get an error and not an ICE for this problematic case.
struct Foo<T = Option<U>, U = bool>;

fn main() {
    let x: Foo;
    //~^ ERROR missing type param `U` in the substitution of `std::option::Option<U>`
}
