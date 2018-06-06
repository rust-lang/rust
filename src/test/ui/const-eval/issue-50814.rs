// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Unsigned {
    const MAX: u8;
}

struct U8(u8);
impl Unsigned for U8 {
    const MAX: u8 = 0xff;
}

struct Sum<A,B>(A,B);

impl<A: Unsigned, B: Unsigned> Unsigned for Sum<A,B> {
    const MAX: u8 = A::MAX + B::MAX;
}

fn foo<T>(_: T) -> &'static u8 {
    &Sum::<U8,U8>::MAX //~ ERROR erroneous constant used
//~| ERROR E0080
}

fn main() {
    foo(0);
}
