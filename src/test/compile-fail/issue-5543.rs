// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

trait Foo { fn foo(&self) {} }
impl Foo for u8 {}

fn main() {
    let r: Box<Foo> = box 5;
    let _m: Box<Foo> = r as Box<Foo>;
    //~^ ERROR `core::marker::Sized` is not implemented for the type `Foo`
}
