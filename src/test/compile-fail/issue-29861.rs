// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait MakeRef<'a> {
    type Ref;
}
impl<'a, T: 'a> MakeRef<'a> for T {
    type Ref = &'a T;
}

pub trait MakeRef2 {
    type Ref2;
}
impl<'a, T: 'a> MakeRef2 for T {
//~^ ERROR the lifetime parameter `'a` is not constrained
    type Ref2 = <T as MakeRef<'a>>::Ref;
}

fn foo() -> <String as MakeRef2>::Ref2 { &String::from("foo") }

fn main() {
    println!("{}", foo());
}
