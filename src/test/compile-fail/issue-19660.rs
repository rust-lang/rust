// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: requires `copy` lang_item

#![feature(lang_items, start, no_core, primitive_type)]
#![no_core]

#[lang = "sized"]
trait Sized { }
#[primitive_type] type isize = isize;
#[primitive_type] type u8 = u8;

struct S;

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    let _ = S;
    0
}
