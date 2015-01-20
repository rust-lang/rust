// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro_crate_test.rs
// ignore-stage1

#![feature(plugin)]

#[plugin] #[no_link]
extern crate macro_crate_test;

// The duplicate macro will create a copy of the item with the given identifier
#[duplicate(MyCopy)]
struct MyStruct {
    number: i32
}

trait TestTrait {
    #[duplicate(TestType2)]
    type TestType;

    #[duplicate(required_fn2)]
    fn required_fn(&self);

    #[duplicate(provided_fn2)]
    fn provided_fn(&self) { }
}

impl TestTrait for MyStruct {
    #[duplicate(TestType2)]
    type TestType = f64;

    #[duplicate(required_fn2)]
    fn required_fn(&self) { }
}

fn main() {
    let s = MyStruct { number: 42 };
    s.required_fn();
    s.required_fn2();
    s.provided_fn();
    s.provided_fn2();

    let s = MyCopy { number: 42 };
}
