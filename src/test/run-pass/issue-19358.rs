// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait { fn dummy(&self) { } }

#[derive(Debug)]
struct Foo<T: Trait> {
    foo: T,
}

#[derive(Debug)]
struct Bar<T> where T: Trait {
    bar: T,
}

impl Trait for int {}

fn main() {
    let a = Foo { foo: 12 };
    let b = Bar { bar: 12 };
    println!("{:?} {:?}", a, b);
}
