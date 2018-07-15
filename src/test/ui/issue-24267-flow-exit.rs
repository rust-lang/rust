// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that we reject code when a nonlocal exit (`break`,
// `continue`) causes us to pop over a needed assignment.

pub fn main() {
    foo1();
    foo2();
}

pub fn foo1() {
    let x: i32;
    loop { x = break; }
    println!("{}", x); //~ ERROR use of possibly uninitialized variable: `x`
}

pub fn foo2() {
    let x: i32;
    for _ in 0..10 { x = continue; }
    println!("{}", x); //~ ERROR use of possibly uninitialized variable: `x`
}
