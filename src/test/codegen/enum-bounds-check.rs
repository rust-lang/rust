// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -O

#![crate_type = "lib"]

pub enum Foo {
    A, B
}

// CHECK-LABEL: @lookup
#[no_mangle]
pub fn lookup(buf: &[u8; 2], f: Foo) -> u8 {
    // CHECK-NOT: panic_bounds_check
    buf[f as usize]
}
