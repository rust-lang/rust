// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait Foo {
    type Out: Sized;
}

impl Foo for String {
    type Out = String;
}

trait Bar: Foo {
    const FROM: Self::Out;
}

impl<T: Foo> Bar for T {
    const FROM: &'static str = "foo";
    //~^ ERROR the trait bound `T: Foo` is not satisfied [E0277]
}

fn main() {}
