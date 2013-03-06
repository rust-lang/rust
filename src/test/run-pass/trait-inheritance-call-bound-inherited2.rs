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
trait Baz : Bar { fn h() -> int; }

struct A { x: int }

impl Foo for A { fn f() -> int { 10 } }
impl Bar for A { fn g() -> int { 20 } }
impl Baz for A { fn h() -> int { 30 } }

// Call a function on Foo, given a T: Baz,
// which is inherited via Bar
fn gg<T:Baz>(a: &T) -> int {
    a.f()
}

pub fn main() {
    let a = &A { x: 3 };
    fail_unless!(gg(a) == 10);
}

