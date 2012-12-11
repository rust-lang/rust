// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[forbid(non_camel_case_types)];

struct foo { //~ ERROR type, variant, or trait should have a camel case identifier
    bar: int,
}

enum foo2 { //~ ERROR type, variant, or trait should have a camel case identifier
    Bar
}

struct foo3 { //~ ERROR type, variant, or trait should have a camel case identifier
    bar: int
}

type foo4 = int; //~ ERROR type, variant, or trait should have a camel case identifier

enum Foo5 {
    bar //~ ERROR type, variant, or trait should have a camel case identifier
}

trait foo6 { //~ ERROR type, variant, or trait should have a camel case identifier
}

fn main() { }