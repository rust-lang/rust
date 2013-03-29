// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

struct X { x: int }

fn f1(a: &mut X, b: &mut int, +c: int) -> int {
    let r = a.x + *b + c;
    a.x = 0;
    *b = 10;
    return r;
}

fn f2(a: int, f: &fn(int)) -> int { f(1); return a; }

pub fn main() {
    let mut a = X {x: 1}, b = 2, c = 3;
    assert!((f1(&mut a, &mut b, c) == 6));
    assert!((a.x == 0));
    assert!((b == 10));
    assert!((f2(a.x, |x| a.x = 50) == 0));
    assert!((a.x == 50));
}
