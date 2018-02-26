// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![deny(non_upper_case_globals)]

static foo: isize = 1;
//~^ ERROR static variable `foo` should have an upper case name such as `FOO`

static mut bar: isize = 1;
//~^ ERROR static variable `bar` should have an upper case name such as `BAR`

fn main() {
    const b: usize = 1;
    //~^ ERROR constant `b` should have an upper case name such as `B`
}

trait Foo {
    const camelCase: usize;
    //~^ ERROR associated constant `camelCase` should have an upper case name such as `CAMEL_CASE`
}

impl Foo for i32 {
    const camelCase: usize = 2;
    //~^ ERROR associated constant `camelCase` should have an upper case name such as `CAMEL_CASE`
}
