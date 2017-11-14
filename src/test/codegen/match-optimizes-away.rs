// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// no-system-llvm
// compile-flags: -O
#![crate_type="lib"]

pub enum Three { First, Second, Third }
use Three::*;

pub enum Four { First, Second, Third, Fourth }
use Four::*;

#[no_mangle]
pub fn three_valued(x: Three) -> Three {
    // CHECK-LABEL: @three_valued
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i8 %0
    match x {
        First => First,
        Second => Second,
        Third => Third,
    }
}

#[no_mangle]
pub fn four_valued(x: Four) -> Four {
    // CHECK-LABEL: @four_valued
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i8 %0
    match x {
        First => First,
        Second => Second,
        Third => Third,
        Fourth => Fourth,
    }
}
