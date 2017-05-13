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

#![deny(dead_code)]

//~ TRANS_ITEM drop-glue tuple_drop_glue::Dropped[0]
//~ TRANS_ITEM drop-glue-contents tuple_drop_glue::Dropped[0]
struct Dropped;

impl Drop for Dropped {
    //~ TRANS_ITEM fn tuple_drop_glue::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

//~ TRANS_ITEM fn tuple_drop_glue::main[0]
fn main() {
    //~ TRANS_ITEM drop-glue (u32, tuple_drop_glue::Dropped[0])
    let x = (0u32, Dropped);

    //~ TRANS_ITEM drop-glue (i16, (tuple_drop_glue::Dropped[0], bool))
    //~ TRANS_ITEM drop-glue (tuple_drop_glue::Dropped[0], bool)
    let x = (0i16, (Dropped, true));
}
