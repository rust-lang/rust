// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the lifetime from the enclosing `&` is "inherited"
// through the `MyBox` struct.

#![allow(dead_code)]
#![feature(rustc_error)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a MyBox<Test>,
    u: &'a MyBox<Test+'a>,
}

struct MyBox<T:?Sized> {
    b: Box<T>
}

fn c<'a>(t: &'a MyBox<Test+'a>, mut ss: SomeStruct<'a>) {
    ss.t = t; //~ ERROR mismatched types
}

fn main() {
}
