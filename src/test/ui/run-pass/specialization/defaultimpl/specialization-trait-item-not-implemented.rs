// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that we can combine a default impl that supplies one method with a
// full impl that supplies the other, and they can invoke one another.

#![feature(specialization)]

trait Foo {
    fn foo_one(&self) -> &'static str;
    fn foo_two(&self) -> &'static str;
    fn foo_three(&self) -> &'static str;
}

struct MyStruct;

default impl<T> Foo for T {
    fn foo_one(&self) -> &'static str {
        self.foo_three()
    }
}

impl Foo for MyStruct {
    fn foo_two(&self) -> &'static str {
        self.foo_one()
    }

    fn foo_three(&self) -> &'static str {
        "generic"
    }
}

fn main() {
    assert!(MyStruct.foo_two() == "generic");
}
