// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait MyTrait {
    fn foo(&self);
}

struct MyStruct {
    x: int,
    y: int,
}

impl MyTrait for MyStruct {
    fn foo(&self) {
        println!("hello world!");
    }
}

fn bar(x: &MyTrait + Send + Share) {
    x.foo();
}

fn main() {
    let traity: Box<MyTrait + Send + Share> = box MyStruct {
        x: 1,
        y: 2,
    };
    bar(traity);
}

