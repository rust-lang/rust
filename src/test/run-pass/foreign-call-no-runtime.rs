// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cast;
use std::libc;
use std::unstable::run_in_bare_thread;

#[link(name = "rustrt")]
extern {
    fn rust_dbg_call(cb: extern "C" fn(libc::uintptr_t),
                     data: libc::uintptr_t) -> libc::uintptr_t;
}

pub fn main() {
    unsafe {
        run_in_bare_thread(proc() {
            let i = &100;
            rust_dbg_call(callback, cast::transmute(i));
        });
    }
}

extern fn callback(data: libc::uintptr_t) {
    unsafe {
        let data: *int = cast::transmute(data);
        assert_eq!(*data, 100);
    }
}
