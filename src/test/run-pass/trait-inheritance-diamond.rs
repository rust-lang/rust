// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// B and C both require A, so D does as well, twice, but that's just fine

trait A { fn a(&self) -> int; }
trait B: A { fn b(&self) -> int; }
trait C: A { fn c(&self) -> int; }
trait D: B + C { fn d(&self) -> int; }

struct S { bogus: () }

impl A for S { fn a(&self) -> int { 10 } }
impl B for S { fn b(&self) -> int { 20 } }
impl C for S { fn c(&self) -> int { 30 } }
impl D for S { fn d(&self) -> int { 40 } }

fn f<T:D>(x: &T) {
    assert_eq!(x.a(), 10);
    assert_eq!(x.b(), 20);
    assert_eq!(x.c(), 30);
    assert_eq!(x.d(), 40);
}

pub fn main() {
    let value = &S { bogus: () };
    f(value);
}
