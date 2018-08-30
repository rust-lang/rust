// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

impl A for i32 {
    type Foo = u32;
}
impl B for u32 {
    const BAR: i32 = 0;
}

trait A {
    type Foo: B;
}

trait B {
    const BAR: i32;
}

fn generic<T: A>() {
    // This panics if the universal function call syntax is used as well
    println!("{}", T::Foo::BAR);
}

fn main() {}
