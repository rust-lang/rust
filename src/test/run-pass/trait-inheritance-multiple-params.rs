// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait A { fn a(&self) -> int; }
trait B: A { fn b(&self) -> int; }
trait C: A { fn c(&self) -> int; }

struct S { bogus: () }

impl A for S { fn a(&self) -> int { 10 } }
impl B for S { fn b(&self) -> int { 20 } }
impl C for S { fn c(&self) -> int { 30 } }

// Multiple type params, multiple levels of inheritance
fn f<X:A,Y:B,Z:C>(x: &X, y: &Y, z: &Z) {
    assert_eq!(x.a(), 10);
    assert_eq!(y.a(), 10);
    assert_eq!(y.b(), 20);
    assert_eq!(z.a(), 10);
    assert_eq!(z.c(), 30);
}

pub fn main() {
    let s = &S { bogus: () };
    f(s, s, s);
}
