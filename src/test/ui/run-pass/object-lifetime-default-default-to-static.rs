// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `Box<Test>` is equivalent to `Box<Test+'static>`, both in
// fields and fn arguments.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct {
    t: Box<Test>,
    u: Box<Test+'static>,
}

fn a(t: Box<Test>, mut ss: SomeStruct) {
    ss.t = t;
}

fn b(t: Box<Test+'static>, mut ss: SomeStruct) {
    ss.t = t;
}

fn c(t: Box<Test>, mut ss: SomeStruct) {
    ss.u = t;
}

fn d(t: Box<Test+'static>, mut ss: SomeStruct) {
    ss.u = t;
}

fn main() {
}
