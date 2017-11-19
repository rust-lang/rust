// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

struct S<X, Y> {
    x: X,
    y: Y,
}

fn main() {
    let x: &&Box<i32>;
    let _y = &**x; //[ast]~ ERROR use of possibly uninitialized variable: `**x` [E0381]
                   //[mir]~^ [E0381]

    let x: &&S<i32, i32>;
    let _y = &**x; //[ast]~ ERROR use of possibly uninitialized variable: `**x` [E0381]
                   //[mir]~^ [E0381]

    let x: &&i32;
    let _y = &**x; //[ast]~ ERROR use of possibly uninitialized variable: `**x` [E0381]
                   //[mir]~^ [E0381]


    let mut a: S<i32, i32>;
    a.x = 0;
    let _b = &a.x; //[ast]~ ERROR use of possibly uninitialized variable: `a.x` [E0381]
                   // (deliberately *not* an error under MIR-borrowck)

    let mut a: S<&&i32, &&i32>;
    a.x = &&0;
    let _b = &**a.x; //[ast]~ ERROR use of possibly uninitialized variable: `**a.x` [E0381]
                     // (deliberately *not* an error under MIR-borrowck)


    let mut a: S<i32, i32>;
    a.x = 0;
    let _b = &a.y; //[ast]~ ERROR use of possibly uninitialized variable: `a.y` [E0381]
                   //[mir]~^ ERROR [E0381]

    let mut a: S<&&i32, &&i32>;
    a.x = &&0;
    let _b = &**a.y; //[ast]~ ERROR use of possibly uninitialized variable: `**a.y` [E0381]
                     //[mir]~^ ERROR [E0381]
}
