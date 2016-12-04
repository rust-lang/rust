// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::Path;

trait Foo {
    fn bar(&self);
}

fn some_func<T: Foo>(foo: T) {
    foo.bar();
}

fn f(p: Path) { }
//~^ ERROR the trait bound `[u8]: std::marker::Sized` is not satisfied in `std::path::Path`
//~| NOTE within `std::path::Path`, the trait `std::marker::Sized` is not implemented for `[u8]`
//~| NOTE `[u8]` does not have a constant size known at compile-time
//~| NOTE required because it appears within the type `std::path::Path`
//~| NOTE all local variables must have a statically known size

fn main() {
    some_func(5i32);
    //~^ ERROR the trait bound `i32: Foo` is not satisfied
    //~| NOTE the trait `Foo` is not implemented for `i32`
    //~| NOTE required by `some_func`
}
