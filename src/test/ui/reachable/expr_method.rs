// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(never_type)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]

struct Foo;

impl Foo {
    fn foo(&self, x: !, y: usize) { }
    fn bar(&self, x: !) { }
}

fn a() {
    // the `22` is unreachable:
    Foo.foo(return, 22); //~ ERROR unreachable
}

fn b() {
    // the call is unreachable:
    Foo.bar(return); //~ ERROR unreachable
}

fn main() { }
