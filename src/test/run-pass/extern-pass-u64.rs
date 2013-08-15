// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a call to a function that takes/returns a u64.

extern {
    pub fn rust_dbg_extern_identity_u64(v: u64) -> u64;
}

#[fixed_stack_segment] #[inline(never)]
pub fn main() {
    unsafe {
        assert_eq!(22_u64, rust_dbg_extern_identity_u64(22_u64));
    }
}
