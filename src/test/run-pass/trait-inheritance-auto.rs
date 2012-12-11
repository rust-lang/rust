// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that this impl turns A into a Quux, because
// A is already a Foo Bar Baz
impl<T: Foo Bar Baz> T: Quux { }

trait Foo { fn f() -> int; }
trait Bar { fn g() -> int; }
trait Baz { fn h() -> int; }

trait Quux: Foo Bar Baz { }

struct A { x: int }

impl A : Foo { fn f() -> int { 10 } }
impl A : Bar { fn g() -> int { 20 } }
impl A : Baz { fn h() -> int { 30 } }

fn f<T: Quux>(a: &T) {
    assert a.f() == 10;
    assert a.g() == 20;
    assert a.h() == 30;
}

fn main() {
    let a = &A { x: 3 };
    f(a);
}

