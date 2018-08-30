// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_intrinsics)]

#[repr(i8)]
pub enum Enum {
    VariantA,
    VariantB,
}

fn make_b() -> Enum { Enum::VariantB }

fn main() {
    assert_eq!(1, make_b() as i8);
    assert_eq!(1, make_b() as u8);
    assert_eq!(1, make_b() as i32);
    assert_eq!(1, make_b() as u32);
    assert_eq!(1, unsafe { std::intrinsics::discriminant_value(&make_b()) });
}
