// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-10028.rs

extern crate issue10028 = "issue-10028";

use issue10028::ZeroLengthThingWithDestructor;

struct Foo {
    zero_length_thing: ZeroLengthThingWithDestructor
}

fn make_foo() -> Foo {
    Foo { zero_length_thing: ZeroLengthThingWithDestructor::new() }
}

fn main() {
    let _f:Foo = make_foo();
}
