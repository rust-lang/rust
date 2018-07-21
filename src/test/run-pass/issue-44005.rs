// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Foo<'a> {
    type Bar;
    fn foo(&'a self) -> Self::Bar;
}

impl<'a, 'b, T: 'a> Foo<'a> for &'b T {
    type Bar = &'a T;
    fn foo(&'a self) -> &'a T {
        self
    }
}

pub fn uncallable<T, F>(x: T, f: F)
    where T: for<'a> Foo<'a>,
          F: for<'a> Fn(<T as Foo<'a>>::Bar)
{
    f(x.foo());
}

pub fn catalyst(x: &i32) {
    broken(x, |_| {})
}

pub fn broken<F: Fn(&i32)>(x: &i32, f: F) {
    uncallable(x, |y| f(y));
}

fn main() { }

