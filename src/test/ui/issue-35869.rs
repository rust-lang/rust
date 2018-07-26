// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn foo(_: fn(u8) -> ());
    fn bar(_: Option<u8>);
    fn baz(_: (u8, u16));
    fn qux() -> u8;
}

struct Bar;

impl Foo for Bar {
    fn foo(_: fn(u16) -> ()) {}
    //~^ ERROR method `foo` has an incompatible type for trait
    fn bar(_: Option<u16>) {}
    //~^ ERROR method `bar` has an incompatible type for trait
    fn baz(_: (u16, u16)) {}
    //~^ ERROR method `baz` has an incompatible type for trait
    fn qux() -> u16 { 5u16 }
    //~^ ERROR method `qux` has an incompatible type for trait
}

fn main() {}
