// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// WONTFIX(#20184) Needs landing pads (not present in stage1) or the compiler hangs.
// ignore-stage1
// compile-flags: -C codegen-units=2
// error-pattern: build without -C codegen-units for more exact errors
// ignore-emscripten

#![feature(asm)]

fn main() {
    unsafe {
        asm!("nowayisthisavalidinstruction");
    }
}
