// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// As in `escape-upvar-ref.rs`, test closure that:
//
// - captures a variable `y`
// - stores reference to `y` into another, longer-lived spot
//
// except that the closure does so via a second closure.

// compile-flags:-Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

#[rustc_regions]
fn test() {
    let x = 44;
    let mut p = &x;

    {
        let y = 22;

        let mut closure = || { //~ ERROR `y` does not live long enough [E0597]
            let mut closure1 = || p = &y;
            closure1();
        };

        closure();
    }

    deref(p);
}

fn deref(_p: &i32) { }

fn main() { }
