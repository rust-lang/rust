// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we report an error if the trait ref in a qualified type
// uses invalid type arguments.

trait Foo<T> {
    type Bar;
    fn get_bar(&self) -> Self::Bar;
}

fn f<T:Foo<isize>>(t: &T) {
    let u: <T as Foo<usize>>::Bar = t.get_bar();
    //~^ ERROR the trait `Foo<usize>` is not implemented for the type `T`
}

fn main() { }
