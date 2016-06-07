// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_intrinsics)]

struct NT(str);
struct DST { a: u32, b: str }

fn main() {
    // type_name should support unsized types
    assert_eq!(unsafe {(
        // Slice
        std::intrinsics::type_name::<[u8]>(),
        // str
        std::intrinsics::type_name::<str>(),
        // Trait
        std::intrinsics::type_name::<Send>(),
        // Newtype
        std::intrinsics::type_name::<NT>(),
        // DST
        std::intrinsics::type_name::<DST>()
    )}, ("[u8]", "str", "std::marker::Send", "NT", "DST"));
}
