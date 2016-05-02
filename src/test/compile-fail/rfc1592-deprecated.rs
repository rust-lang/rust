// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

#[deny(warnings)] trait Foo { fn foo(&self) -> (Self, Self); }
//~^ ERROR the trait bound `Self: std::marker::Sized` is not satisfied
//~| WARNING hard error

impl<T: Copy> Foo for T {
    fn foo(&self) -> (Self, Self) {
        (*self, *self)
    }
}

#[deny(warnings)]
fn main() {
    assert_eq!((11).foo(), (11, 11));

    let junk: Box<fmt::Debug+Sized> = Box::new(42);
    //~^ ERROR the trait cannot require that `Self : Sized`
    //~| WARNING hard error
    let f = format!("{:?}", junk);
    assert_eq!(f, "42");
}
