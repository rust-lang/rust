// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "externcallback",
       vers = "0.1")];

#[crate_type = "lib"];

use std::libc;

pub mod rustrt {
    use std::libc;

    extern {
        pub fn rust_dbg_call(cb: *u8, data: libc::uintptr_t)
                             -> libc::uintptr_t;
    }
}

pub fn fact(n: uint) -> uint {
    unsafe {
        info!("n = %?", n);
        rustrt::rust_dbg_call(cb, n)
    }
}

pub extern fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
    if data == 1u {
        data
    } else {
        fact(data - 1u) * data
    }
}
