// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(asm)]
#![feature(lang_items)]
#![feature(no_core)]
#![no_core]
#![no_main]

#[no_mangle]
pub fn _start() -> ! {
    unsafe {
        asm!("bkpt");
    }

    loop {}
}

#[allow(private_no_mangle_fns)]
#[no_mangle]
fn __aeabi_unwind_cpp_pr0() {}

#[lang = "copy"]
trait Copy {}

#[lang = "sized"]
trait Sized {}
