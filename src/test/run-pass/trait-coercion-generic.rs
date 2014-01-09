// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

use std::io::println;

trait Trait<T> {
    fn f(&self, x: T);
}

struct Struct {
    x: int,
    y: int,
}

impl Trait<&'static str> for Struct {
    fn f(&self, x: &'static str) {
        println!("Hi, {}!", x);
    }
}

fn f(x: @Trait<&'static str>) {
    x.f("Sue");
}

pub fn main() {
    let a = Struct { x: 1, y: 2 };
    let b: @Trait<&'static str> = @a;
    b.f("Fred");
    let c: ~Trait<&'static str> = ~a;
    c.f("Mary");
    let d: &Trait<&'static str> = &a;
    d.f("Joe");
    f(@a);
}

