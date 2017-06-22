// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]
#![feature(rustc_attrs, thread_local, lang_items)]

#[lang = "panic_fmt"] fn panic_fmt() -> ! { loop {} }
#[lang = "eh_personality"] extern fn eh_personality() {}

pub struct BB;

#[thread_local]
static mut KEY: Key = Key {
    inner: BB,
    dtor_running: false,
};

pub unsafe fn set() -> Option<&'static BB> {
    if KEY.dtor_running {
        return None
    }
    Some(&KEY.inner)
}

pub struct Key {
    inner: BB,
    dtor_running: bool,
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
}
