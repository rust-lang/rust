// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -O
#![crate_type="lib"]
#![feature(maybe_uninit)]

use std::mem::MaybeUninit;

// Boxing a `MaybeUninit` value should not copy junk from the stack
#[no_mangle]
pub fn box_uninitialized() -> Box<MaybeUninit<usize>> {
    // CHECK-LABEL: @box_uninitialized
    // CHECK-NOT: store
    Box::new(MaybeUninit::uninitialized())
}
