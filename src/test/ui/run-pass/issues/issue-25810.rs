// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let x = X(15);
    let y = x.foo();
    println!("{:?}",y);
}

trait Foo
    where for<'a> &'a Self: Bar
{
    fn foo<'a>(&'a self) -> <&'a Self as Bar>::Output;
}

trait Bar {
    type Output;
}

struct X(i32);

impl<'a> Bar for &'a X {
    type Output = &'a i32;
}

impl Foo for X {
    fn foo<'a>(&'a self) -> <&'a Self as Bar>::Output {
        &self.0
    }
}
