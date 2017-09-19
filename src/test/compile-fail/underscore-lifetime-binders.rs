// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(underscore_lifetimes)]

struct Foo<'a>(&'a u8);

fn foo<'_> //~ ERROR invalid lifetime parameter name: `'_`
(_: Foo<'_>) {}

trait Meh<'a> {}
impl<'a> Meh<'a> for u8 {}

fn meh() -> Box<for<'_> Meh<'_>> //~ ERROR invalid lifetime parameter name: `'_`
//~^ ERROR missing lifetime specifier
//~^^ ERROR missing lifetime specifier
{
  Box::new(5u8)
}

fn main() {
    let x = 5;
    foo(Foo(&x));
    let _ = meh();
}
