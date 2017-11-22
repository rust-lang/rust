// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Coverflow-checks=off

#![allow(exceeding_bitshifts)]
#![feature(generators, generator_trait)]

use std::ops::Generator;

#[deny(const_err)]
fn a() -> u8 {
    [b'a'][1]
}

#[allow(const_err)]
fn b() -> u8 {
    [b'b'][1]
}

fn c() -> u8 {
    [b'c'][1]
}

fn d() -> u8 {
    [b'd'][1]
}

// Make sure the const_err are only emitted **once**
fn check_emit_only_once() {
    let _ = 200u8 + 200u8;
    let _ = 23u8 << 19;
    let _ = 17 / 0;
}

#[allow(const_err)]
mod m {
    pub fn d() -> u8 {
        [b'D'][1]
    }

    #[deny(const_err)]
    pub fn e() -> u8 {
        [b'E'][1]
    }
}

fn main() {
    a();
    b();
    c();
    d();
    m::d();
    m::e();
    check_emit_only_once();
}
