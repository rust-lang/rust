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

trait Trait {
    fn f(&self);
}

struct Struct {
    x: int,
    y: int,
}

impl Trait for Struct {
    fn f(&self) {
        println!("Hi!");
    }
}

fn f(x: @Trait) {
    x.f();
}

pub fn main() {
    let a = Struct { x: 1, y: 2 };
    let b: @Trait = @a;
    b.f();
    let c: ~Trait = ~a;
    c.f();
    let d: &Trait = &a;
    d.f();
    f(@a);
}

