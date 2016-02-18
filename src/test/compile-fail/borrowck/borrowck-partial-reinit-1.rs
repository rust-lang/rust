// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Test;

struct Test2 {
    b: Option<Test>,
}

struct Test3(Option<Test>);

impl Drop for Test {
    fn drop(&mut self) {
        println!("dropping!");
    }
}

impl Drop for Test2 {
    fn drop(&mut self) {}
}

impl Drop for Test3 {
    fn drop(&mut self) {}
}

fn stuff() {
    let mut t = Test2 { b: None };
    let u = Test;
    drop(t);
    t.b = Some(u);
    //~^ ERROR partial reinitialization of uninitialized structure `t`

    let mut t = Test3(None);
    let u = Test;
    drop(t);
    t.0 = Some(u);
    //~^ ERROR partial reinitialization of uninitialized structure `t`
}

fn main() {
    stuff()
}
