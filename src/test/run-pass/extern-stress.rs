// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This creates a bunch of yielding tasks that run concurrently
// while holding onto C stacks

mod rustrt {
    pub extern {
        pub fn rust_dbg_call(cb: *u8, data: libc::uintptr_t)
                          -> libc::uintptr_t;
    }
}

extern fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
    if data == 1u {
        data
    } else {
        task::yield();
        count(data - 1u) + count(data - 1u)
    }
}

fn count(n: uint) -> uint {
    unsafe {
        rustrt::rust_dbg_call(cb, n)
    }
}

pub fn main() {
    for iter::repeat(100u) {
        do task::spawn {
            assert!(count(5u) == 16u);
        };
    }
}
