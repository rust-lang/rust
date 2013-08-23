// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Constants (static variables) can be used to match in patterns, but mutable
// statics cannot. This ensures that there's some form of error if this is
// attempted.

use std::libc;

#[nolink]
extern {
    static mut rust_dbg_static_mut: libc::c_int;
    pub fn rust_dbg_static_mut_check_four();
}

unsafe fn static_bound(_: &'static libc::c_int) {}

fn static_bound_set(a: &'static mut libc::c_int) {
    *a = 3;
}

#[fixed_stack_segment] #[inline(never)]
unsafe fn run() {
    assert!(rust_dbg_static_mut == 3);
    rust_dbg_static_mut = 4;
    assert!(rust_dbg_static_mut == 4);
    rust_dbg_static_mut_check_four();
    rust_dbg_static_mut += 1;
    assert!(rust_dbg_static_mut == 5);
    rust_dbg_static_mut *= 3;
    assert!(rust_dbg_static_mut == 15);
    rust_dbg_static_mut = -3;
    assert!(rust_dbg_static_mut == -3);
    static_bound(&rust_dbg_static_mut);
    static_bound_set(&mut rust_dbg_static_mut);
}

pub fn main() {
    unsafe { run() }
}
