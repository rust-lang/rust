// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn f(x: Box<int>) {
    let y: &int = &*x;
    println!("{}", *x);
    println!("{}", *y);
}

trait Trait {
    fn printme(&self);
}

struct Struct;

impl Trait for Struct {
    fn printme(&self) {
        println!("hello world!");
    }
}

fn g(x: Box<Trait>) {
    x.printme();
    let y: &Trait = &*x;
    y.printme();
}

fn main() {
    f(box 1234);
    g(box Struct as Box<Trait>);
}

