// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn reproduce<T>(t: T) -> @fn() -> T {
    let result: @fn() -> T = || t;
    result
}

fn main() {
    // type of x is the variable X,
    // with the lower bound @mut int
    let x = @mut 3;

    // type of r is @fn() -> X
    let r = reproduce(x);

    // Requires that X be a subtype of
    // @mut int.
    let f: @mut int = r();

    // Bad.
    let h: @int = r(); //~ ERROR (values differ in mutability)
}
