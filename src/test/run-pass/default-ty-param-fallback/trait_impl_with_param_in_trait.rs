// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_parameter_fallback)]

trait A<T = Self> {
    fn a(t: &T) -> Self;
}

trait B<T = Self> {
    fn b(&self) -> T;
}

impl<T = X> B<T> for X
    where T: A<X>
{
    fn b(&self) -> T {
        T::a(self)
    }
}

struct X(u8);

impl A for X {
    fn a(x: &X) -> X {
        X(x.0)
    }
}

fn g(x: &X) {
    // Bug: replacing with `let x = X::b(x)` will cause `x` to be unknown in `x.0`.
    let x = <X as B>::b(x);
    x.0;
}

fn main() {
    g(&X(0));
}
