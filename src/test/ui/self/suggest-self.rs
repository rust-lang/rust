// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    x: i32,
}

impl Foo {
    fn foo(&self) -> i32 {
        this.x
        //~^ ERROR cannot find value `this` in this scope
    }

    fn bar(&self) -> i32 {
        this.foo()
        //~^ ERROR cannot find value `this` in this scope
    }

    fn baz(&self) -> i32 {
        my.bar()
        //~^ ERROR cannot find value `this` in this scope
    }
}

fn main() {
    let this = vec![1, 2, 3];
    let my = vec![1, 2, 3];
    let len = this.len();
    let len = my.len();
}

