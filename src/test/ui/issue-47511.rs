// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #47511: anonymous lifetimes can appear
// unconstrained in a return type, but only if they appear just once
// in the input, as the input to a projection.

fn f(_: X) -> X {
    //~^ ERROR return type references an anonymous lifetime
    unimplemented!()
}

fn g<'a>(_: X<'a>) -> X<'a> {
    //~^ ERROR return type references lifetime `'a`, which is not constrained
    unimplemented!()
}

type X<'a> = <&'a () as Trait>::Value;

trait Trait {
    type Value;
}

impl<'a> Trait for &'a () {
    type Value = ();
}

fn main() {}
