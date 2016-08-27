// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]

trait Foo {
    fn foo(fn(u8) -> ()); //~ NOTE type in trait
    fn bar(Option<u8>); //~ NOTE type in trait
    fn baz((u8, u16)); //~ NOTE type in trait
    fn qux() -> u8; //~ NOTE type in trait
}

struct Bar;

impl Foo for Bar {
    fn foo(_: fn(u16) -> ()) {}
    //~^ ERROR method `foo` has an incompatible type for trait
    //~| NOTE expected u8
    fn bar(_: Option<u16>) {}
    //~^ ERROR method `bar` has an incompatible type for trait
    //~| NOTE expected u8
    fn baz(_: (u16, u16)) {}
    //~^ ERROR method `baz` has an incompatible type for trait
    //~| NOTE expected u8
    fn qux() -> u16 { 5u16 }
    //~^ ERROR method `qux` has an incompatible type for trait
    //~| NOTE expected u8
}

fn main() {}
