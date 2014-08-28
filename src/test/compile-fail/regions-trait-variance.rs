// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #12470.

trait X {
    fn get_i(&self) -> int;
}

struct B {
    i: int
}

impl X for B {
    fn get_i(&self) -> int {
        self.i
    }
}

impl Drop for B {
    fn drop(&mut self) {
        println!("drop");
    }
}

struct A<'r> {
    p: &'r X+'r
}

fn make_a(p:&X) -> A {
    A{p:p}
}

fn make_make_a<'a>() -> A<'a> {
    let b: Box<B> = box B {
        i: 1,
    };
    let bb: &B = &*b; //~ ERROR `*b` does not live long enough
    make_a(bb)
}

fn main() {
    let a = make_make_a();
    println!("{}", a.p.get_i());
}
