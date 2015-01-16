// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unstable)]

struct NT(str);
struct DST { a: u32, b: str }

fn main() {
    // get_tydesc should support unsized types
    assert!(unsafe {(
        // Slice
        (*std::intrinsics::get_tydesc::<[u8]>()).name,
        // str
        (*std::intrinsics::get_tydesc::<str>()).name,
        // Trait
        (*std::intrinsics::get_tydesc::<Copy>()).name,
        // Newtype
        (*std::intrinsics::get_tydesc::<NT>()).name,
        // DST
        (*std::intrinsics::get_tydesc::<DST>()).name
    )} == ("[u8]", "str", "core::marker::Copy + 'static", "NT", "DST"));
}
