// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(const_err)]

trait Foo {
    const AMT: usize;
}

enum Bar<A, B> {
    First(A),
    Second(B),
}

impl<A: Foo, B: Foo> Foo for Bar<A, B> {
    const AMT: usize = [A::AMT][(A::AMT > B::AMT) as usize];
}

impl Foo for u8 {
    const AMT: usize = 1;
}

impl Foo for u16 {
    const AMT: usize = 2;
}

fn main() {
    println!("{}", <Bar<u16, u8> as Foo>::AMT);
    //~^ ERROR erroneous constant used
    //~| ERROR E0080
}
