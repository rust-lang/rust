// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android
// ignore-arm
// ignore-aarch64
// ignore-s390x
// ignore-emscripten

#![feature(asm, rustc_attrs)]

#[cfg(any(target_arch = "x86",
          target_arch = "x86_64"))]
#[rustc_error]
pub fn main() {
    unsafe {
        // clobber formatted as register input/output
        asm!("xor %eax, %eax" : : : "{eax}");
        //~^ ERROR clobber should not be surrounded by braces
    }
}
