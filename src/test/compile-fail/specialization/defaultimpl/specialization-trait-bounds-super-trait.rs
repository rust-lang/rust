// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: the trait bound `MyStruct: SuperFoo` is not satisfied

#![feature(specialization)]

trait SuperFoo {
    fn super_foo_one(&self) -> &'static str;
    fn super_foo_two(&self) -> &'static str;
}

trait Foo: SuperFoo {
    fn foo(&self) -> &'static str;
}

default impl<T> SuperFoo for T {
    fn super_foo_one(&self) -> &'static str {
        "generic"
    }
}

struct MyStruct;

impl Foo for MyStruct {
    fn foo(&self) -> &'static str {
        "foo"
    }
}

fn main() {
    println!("{:?}", MyStruct.foo());
}