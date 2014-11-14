// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Test {
    a: int,
    b: Option<Box<Test>>,
}

impl Drop for Test {
    fn drop(&mut self) {
        println!("Dropping {}", self.a);
    }
}

fn stuff() {
    let mut t = Test { a: 1, b: None};
    let mut u = Test { a: 2, b: Some(box t)};    
    t.b = Some(box u);
    //~^ ERROR partial reinitialization of uninitialized structure
    println!("done");
}

fn main() {
    stuff();
    println!("Hello, world!")
}

