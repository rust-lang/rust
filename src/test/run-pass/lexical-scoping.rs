// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that items in subscopes can shadow type parameters and local variables (see issue #23880).

#![allow(unused)]
struct Foo<X> { x: Box<X> }
impl<Bar> Foo<Bar> {
    fn foo(&self) {
        type Bar = i32;
        let _: Bar = 42;
    }
}

fn main() {
    let f = 1;
    {
        fn f() {}
        f();
    }
}
