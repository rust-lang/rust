// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(advanced_slice_patterns,)]

fn f<T,>(_: T,) {}

struct Foo<T,>(T);

struct Bar;

impl Bar {
    fn f(_: int,) {}
    fn g(self, _: int,) {}
    fn h(self,) {}
}

enum Baz {
    Qux(int,),
}

#[allow(unused,)]
pub fn main() {
    f::<int,>(0,);
    let (_, _,) = (1, 1,);
    let [_, _,] = [1, 1,];
    let [_, _, .., _,] = [1, 1, 1, 1,];
    let [_, _, _.., _,] = [1, 1, 1, 1,];

    let x: Foo<int,> = Foo::<int,>(1);

    Bar::f(0,);
    Bar.g(0,);
    Bar.h();

    let x = Baz::Qux(1,);
}
