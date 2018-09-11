// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for using an object with an associated type binding as the
// instantiation for a generic type with a bound.


trait SomeTrait {
    type SomeType;

    fn get(&self) -> Self::SomeType;
}

fn get_int<T:SomeTrait<SomeType=i32>+?Sized>(x: &T) -> i32 {
    x.get()
}

impl SomeTrait for i32 {
    type SomeType = i32;
    fn get(&self) -> i32 {
        *self
    }
}

fn main() {
    let x = 22;
    let x1: &SomeTrait<SomeType=i32> = &x;
    let y = get_int(x1);
    assert_eq!(x, y);
}
