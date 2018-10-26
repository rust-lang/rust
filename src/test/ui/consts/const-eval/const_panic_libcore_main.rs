// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "bin"]
#![feature(lang_items)]
#![feature(const_panic)]
#![no_main]
#![no_std]

use core::panic::PanicInfo;

const Z: () = panic!("cheese");
//~^ ERROR any use of this value will cause an error

const Y: () = unreachable!();
//~^ ERROR any use of this value will cause an error

const X: () = unimplemented!();
//~^ ERROR any use of this value will cause an error

#[lang = "eh_personality"]
fn eh() {}
#[lang = "eh_unwind_resume"]
fn eh_unwind_resume() {}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
