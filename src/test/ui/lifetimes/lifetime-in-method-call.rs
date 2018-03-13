// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S;

impl S {
    fn foo(&self, x: &usize) {}
}

fn foo(x: &usize) {}

fn main() {
    'l: loop {
        break 'l;
    }
    'y: loop {
        let x = 3;
        foo(&'a x);
        //~^ ERROR found unexpected lifetime
        break 'y;
    }
    let x = 4;
    let s = S;
    s.foo(&'a mut x);
    //~^ ERROR found unexpected lifetime
}
