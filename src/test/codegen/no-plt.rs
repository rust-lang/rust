// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C relocation-model=pic -Z plt=no

#![crate_type = "lib"]

// We need a function which is normally called through the PLT.
extern "C" {
    // CHECK: Function Attrs: nounwind nonlazybind
    fn getenv(name: *const u8) -> *mut u8;
}

// Ensure the function gets referenced.
pub unsafe fn call_through_plt() -> *mut u8 {
    getenv(b"\0".as_ptr())
}

// Ensure intrinsics also skip the PLT
// CHECK: !"RtLibUseGOT"
