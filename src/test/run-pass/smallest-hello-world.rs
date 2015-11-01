// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Smallest "hello world" with a libc runtime

// pretty-expanded FIXME #23616

#![feature(intrinsics, lang_items, start, no_core, libc)]
#![no_core]

extern crate libc;

extern { fn puts(s: *const u8); }
extern "rust-intrinsic" { fn transmute<T, U>(t: T) -> U; }

#[lang = "eh_personality"] extern fn eh_personality() {}
#[lang = "eh_unwind_resume"] extern fn eh_unwind_resume() {}
#[lang = "panic_fmt"] fn panic_fmt() -> ! { loop {} }
#[no_mangle] pub extern fn rust_eh_register_frames () {}
#[no_mangle] pub extern fn rust_eh_unregister_frames () {}

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    unsafe {
        let (ptr, _): (*const u8, usize) = transmute("Hello!\0");
        puts(ptr);
    }
    return 0;
}

#[cfg(target_os = "android")]
#[link(name="gcc")]
extern { }
