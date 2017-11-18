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

pub enum Three { A, B, C }

pub enum Four { A, B, C, D }

#[no_mangle]
pub fn three_valued(x: Three) -> Three {
    // CHECK-LABEL: @three_valued
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i8 %0
    match x {
        Three::A => Three::A,
        Three::B => Three::B,
        Three::C => Three::C,
    }
}

#[no_mangle]
pub fn four_valued(x: Four) -> Four {
    // CHECK-LABEL: @four_valued
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i8 %0
    match x {
        Four::A => Four::A,
        Four::B => Four::B,
        Four::C => Four::C,
        Four::D => Four::D,
    }
}
