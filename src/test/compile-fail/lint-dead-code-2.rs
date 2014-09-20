// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variable)]
#![deny(dead_code)]

struct Foo;

trait Bar {
    fn bar1(&self);
    fn bar2(&self) {
        self.bar1();
    }
}

impl Bar for Foo {
    fn bar1(&self) {
        live_fn();
    }
}

fn live_fn() {}

fn dead_fn() {} //~ ERROR: function is never used

#[main]
fn dead_fn2() {} //~ ERROR: function is never used

fn used_fn() {}

#[start]
fn start(_: int, _: *const *const u8) -> int {
    used_fn();
    let foo = Foo;
    foo.bar2();
    0
}

// this is not main
fn main() { //~ ERROR: function is never used
    dead_fn();
    dead_fn2();
}
