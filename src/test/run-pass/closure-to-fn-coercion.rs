// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage0: new feature, remove this when SNAP

#![feature(closure_to_fn_coercion)]

const FOO: fn(u8) -> u8 = |v: u8| { v };

const BAR: [fn(&mut u32); 5] = [
    |_: &mut u32| {},
    |v: &mut u32| *v += 1,
    |v: &mut u32| *v += 2,
    |v: &mut u32| *v += 3,
    |v: &mut u32| *v += 4,
];
fn func_specific() -> (fn() -> u32) {
    || return 42
}

fn main() {
    // Items
    assert_eq!(func_specific()(), 42);
    let foo: fn(u8) -> u8 = |v: u8| { v };
    assert_eq!(foo(31), 31);
    // Constants
    assert_eq!(FOO(31), 31);
    let mut a: u32 = 0;
    assert_eq!({ BAR[0](&mut a); a }, 0);
    assert_eq!({ BAR[1](&mut a); a }, 1);
    assert_eq!({ BAR[2](&mut a); a }, 3);
    assert_eq!({ BAR[3](&mut a); a }, 6);
    assert_eq!({ BAR[4](&mut a); a }, 10);
}
