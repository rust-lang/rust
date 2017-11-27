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

trait B<T> {
    fn b(&self) -> T;
}

impl<T = X> B<T> for X
    where T: Default
{
    fn b(&self) -> T {
        T::default()
    }
}

#[derive(Copy, Clone, Default)]
struct X(u8);

fn main() {
    let x = X(0);
    let y = x.b();
    foo(y);
    // Uncommenting the following line makes the fallback fail.
    // Probably we are creating a new inference var,
    // that is carrying over the origin but not the value of the default.
    // x.0;
}

fn foo<T = X>(a: T) -> T {
    a
}
