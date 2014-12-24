// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A;

trait U {}

// impl U for A {}

fn equal<T>(_: &T, _: &T) -> bool where A : U {
    true
}

fn main() {
    equal(&0i, &0i);
    //~^ ERROR the trait `U` is not implemented for the type `A`
}
