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
// access to the variable, whether that mutable access be used
// for direct assignment or for taking mutable ref. Issue #6801.


fn a() {
    let mut x = 3;
    let c1 = || x = 4;
    let c2 = || x = 5; //~ ERROR cannot borrow `x` as mutable more than once
}

fn set(x: &mut int) {
    *x = 4;
}

fn b() {
    let mut x = 3;
    let c1 = || set(&mut x);
    let c2 = || set(&mut x); //~ ERROR cannot borrow `x` as mutable more than once
}

fn c() {
    let mut x = 3;
    let c1 = || x = 5;
    let c2 = || set(&mut x); //~ ERROR cannot borrow `x` as mutable more than once
}

fn d() {
    let mut x = 3;
    let c1 = || x = 5;
    let c2 = || { let _y = || set(&mut x); }; // (nested closure)
    //~^ ERROR cannot borrow `x` as mutable more than once
}

fn g() {
    struct Foo {
        f: Box<int>
    }

    let mut x = box Foo { f: box 3 };
    let c1 = || set(&mut *x.f);
    let c2 = || set(&mut *x.f);
    //~^ ERROR cannot borrow `x` as mutable more than once
}

fn main() {
}
