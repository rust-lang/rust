// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(trait_alias)]

type Foo = std::fmt::Debug;
type Bar = Foo;

fn foo<T: Foo>(v: &T) {
    println!("{:?}", v);
}

pub fn main() {
    foo(&12345);

    let bar1: &Bar = &54321;
    println!("{:?}", bar1);

    let bar2 = Box::new(42) as Box<dyn Foo>;
    println!("{:?}", bar2);
}
