// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variable, dead_code)]
#![deny(unused_lifetimes)]

struct Foo1<'a> { //~ ERROR: unused lifetime in struct
    a: int
}

struct Foo2<'a, 'b> { //~ ERROR: unused lifetime in struct
    a: int,
    b: Option<&'a mut int>
}

struct Foo3;

impl<'a> Foo3 { //~ ERROR: unused lifetime in impl
    fn test1() {}
}

impl<'a, 'c> Foo1<'a> { //~ ERROR: unused lifetime in impl
    fn test2<'a, 'b>(&'a self, x: int) {} //~ ERROR: unused lifetime in method
}

trait Bar<'a> { //~ ERROR: unused lifetime in trait
    fn test3() {}
}

fn test4<'a>() {} //~ ERROR: unused lifetime in function
fn test5<'a>(x: int) {} //~ ERROR: unused lifetime in function
fn test6<'a, 'b>(x: &'a int) {} //~ ERROR: unused lifetime in function
fn test7<'a, 'b, 'c>(x: Option<Foo1<'a>>, y: Option<&'b int>, z: int) {}
//~^ ERROR: unused lifetime in function

// These are here to make sure lifetimes in closures
// are taken into account
fn test8<'a>(x: |int|: 'a -> int) {}
fn test9<'a>(x: |int, &'a int| -> int) {}

fn main() {}

