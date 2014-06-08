// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(non_camel_case_types)]
#![allow(dead_code)]

struct foo { //~ ERROR type `foo` should have a camel case name such as `Foo`
    bar: int,
}

enum foo2 { //~ ERROR type `foo2` should have a camel case name such as `Foo2`
    Bar
}

struct foo3 { //~ ERROR type `foo3` should have a camel case name such as `Foo3`
    bar: int
}

type foo4 = int; //~ ERROR type `foo4` should have a camel case name such as `Foo4`

enum Foo5 {
    bar //~ ERROR variant `bar` should have a camel case name such as `Bar`
}

trait foo6 { //~ ERROR trait `foo6` should have a camel case name such as `Foo6`
}

fn main() { }
