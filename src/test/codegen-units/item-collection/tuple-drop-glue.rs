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

//~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<tuple_drop_glue::Dropped[0]> @@ tuple_drop_glue0[Internal]
struct Dropped;

impl Drop for Dropped {
    //~ TRANS_ITEM fn tuple_drop_glue::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

//~ TRANS_ITEM fn tuple_drop_glue::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<(u32, tuple_drop_glue::Dropped[0])> @@ tuple_drop_glue0[Internal]
    let x = (0u32, Dropped);

    //~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<(i16, (tuple_drop_glue::Dropped[0], bool))> @@ tuple_drop_glue0[Internal]
    //~ TRANS_ITEM fn core::ptr[0]::drop_in_place[0]<(tuple_drop_glue::Dropped[0], bool)> @@ tuple_drop_glue0[Internal]
    let x = (0i16, (Dropped, true));

    0
}
