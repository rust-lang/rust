// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that a user-defined type admits multiple inherent methods
// with the same name, which can be called on values that have a
// precise enough type to allow distinguishing between the methods.


struct Foo<T>(T);

impl Foo<usize> {
    fn bar(&self) -> i32 { self.0 as i32 }
}

impl Foo<isize> {
    fn bar(&self) -> i32 { -(self.0 as i32) }
}

fn main() {
    let foo_u = Foo::<usize>(5);
    assert_eq!(foo_u.bar(), 5);

    let foo_i = Foo::<isize>(3);
    assert_eq!(foo_i.bar(), -3);
}
