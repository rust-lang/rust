// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that two closures cannot simultaneously have mutable
// and immutable access to the variable. Issue #6801.


fn get(x: &int) -> int {
    *x
}

fn set(x: &mut int) {
    *x = 4;
}

fn a() {
    let mut x = 3;
    let c1 = || x = 4;
    let c2 = || x * 5; //~ ERROR cannot borrow `x`
}

fn b() {
    let mut x = 3;
    let c1 = || set(&mut x);
    let c2 = || get(&x); //~ ERROR cannot borrow `x`
}

fn c() {
    let mut x = 3;
    let c1 = || set(&mut x);
    let c2 = || x * 5; //~ ERROR cannot borrow `x`
}

fn d() {
    let mut x = 3;
    let c2 = || x * 5;
    x = 5; //~ ERROR cannot assign
}

fn e() {
    let mut x = 3;
    let c1 = || get(&x);
    x = 5; //~ ERROR cannot assign
}

fn f() {
    let mut x = box 3;
    let c1 = || get(&*x);
    *x = 5; //~ ERROR cannot assign
}

fn g() {
    struct Foo {
        f: Box<int>
    }

    let mut x = box Foo { f: box 3 };
    let c1 = || get(&*x.f);
    *x.f = 5; //~ ERROR cannot assign to `*x.f`
}

fn h() {
    struct Foo {
        f: Box<int>
    }

    let mut x = box Foo { f: box 3 };
    let c1 = || get(&*x.f);
    let c2 = || *x.f = 5; //~ ERROR cannot borrow `x` as mutable
}

fn main() {
}
