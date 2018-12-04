// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]
#![allow(warnings)]

struct Test;

impl Test {
    fn bar(&self, x: &u32) -> &u32 {
        let x = 22;
        &x
//~^ ERROR cannot return reference to local variable
    }
}

fn foo(x: &u32) -> &u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn baz(x: &u32) -> &&u32 {
    let x = 22;
    &&x
//~^ ERROR cannot return value referencing local variable
//~| ERROR cannot return reference to temporary value
}

fn foobazbar<'a>(x: u32, y: &'a u32) -> &'a u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn foobar<'a>(x: &'a u32) -> &'a u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn foobaz<'a, 'b>(x: &'a u32, y: &'b u32) -> &'a u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn foobarbaz<'a, 'b>(x: &'a u32, y: &'b u32, z: &'a u32) -> &'a u32 {
    let x = 22;
    &x
//~^ ERROR cannot return reference to local variable
}

fn main() { }
