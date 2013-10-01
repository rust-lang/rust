// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::libc;
use std::task;

mod rustrt {
    use std::libc;

    extern {
        pub fn rust_dbg_call(cb: extern "C" fn (libc::uintptr_t) -> libc::uintptr_t,
                             data: libc::uintptr_t)
                             -> libc::uintptr_t;
    }
}

extern fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
    if data == 1u {
        data
    } else {
        count(data - 1u) + count(data - 1u)
    }
}

#[fixed_stack_segment] #[inline(never)]
fn count(n: uint) -> uint {
    unsafe {
        task::deschedule();
        rustrt::rust_dbg_call(cb, n)
    }
}

pub fn main() {
    do 10u.times {
        do task::spawn {
            let result = count(5u);
            info2!("result = {}", result);
            assert_eq!(result, 16u);
        };
    }
}
