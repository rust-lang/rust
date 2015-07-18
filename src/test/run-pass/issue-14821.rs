// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait SomeTrait {}
struct Meow;
impl SomeTrait for Meow {}

struct Foo<'a> {
    x: &'a SomeTrait,
    y: &'a SomeTrait,
}

impl<'a> Foo<'a> {
    pub fn new<'b>(x: &'b SomeTrait, y: &'b SomeTrait) -> Foo<'b> { Foo { x: x, y: y } }
}

fn main() {
    let r = Meow;
    let s = Meow;
    let q = Foo::new(&r as &SomeTrait, &s as &SomeTrait);
}
