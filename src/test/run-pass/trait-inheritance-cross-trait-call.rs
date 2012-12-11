// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo { fn f() -> int; }
trait Bar : Foo { fn g() -> int; }

struct A { x: int }

impl A : Foo { fn f() -> int { 10 } }

impl A : Bar {
    // Testing that this impl can call the impl of Foo
    fn g() -> int { self.f() }
}

fn main() {
    let a = &A { x: 3 };
    assert a.g() == 10;
}

