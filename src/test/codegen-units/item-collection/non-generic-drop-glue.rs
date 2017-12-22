// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager
// compile-flags:-Zinline-in-all-cgus

#![deny(dead_code)]
#![feature(start)]

//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<non_generic_drop_glue::StructWithDrop[0]> @@ non_generic_drop_glue0[Internal]
struct StructWithDrop {
    x: i32
}

impl Drop for StructWithDrop {
    //~ TRANS_ITEM fn non_generic_drop_glue::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

struct StructNoDrop {
    x: i32
}

//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<non_generic_drop_glue::EnumWithDrop[0]> @@ non_generic_drop_glue0[Internal]
enum EnumWithDrop {
    A(i32)
}

impl Drop for EnumWithDrop {
    //~ TRANS_ITEM fn non_generic_drop_glue::{{impl}}[1]::drop[0]
    fn drop(&mut self) {}
}

enum EnumNoDrop {
    A(i32)
}

//~ TRANS_ITEM fn non_generic_drop_glue::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _ = StructWithDrop { x: 0 }.x;
    let _ = StructNoDrop { x: 0 }.x;
    let _ = match EnumWithDrop::A(0) {
        EnumWithDrop::A(x) => x
    };
    let _ = match EnumNoDrop::A(0) {
        EnumNoDrop::A(x) => x
    };

    0
}
