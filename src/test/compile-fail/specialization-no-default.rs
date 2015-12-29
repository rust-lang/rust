// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn foo(&self);
    fn bar(&self);
}

impl<T> Foo for T {
    fn foo(&self) {}
    fn bar(&self) {}
}

impl Foo for u8 {}
impl Foo for u16 {
    fn foo(&self) {} //~ ERROR E0520
}
impl Foo for u32 {
    fn bar(&self) {} //~ ERROR E0520
}

trait Bar {
    type T;
}

impl<T> Bar for T {
    type T = u8;
}

impl Bar for u8 {
    type T = (); //~ ERROR E0520
}

fn main() {}
