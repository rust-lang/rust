// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

mod libc {
    use std::libc::{c_char, size_t};

    #[nolink]
    extern {
        #[link_name = "strlen"]
        pub fn my_strlen(str: *c_char) -> size_t;
    }
}

#[fixed_stack_segment] #[inline(never)]
fn strlen(str: ~str) -> uint {
    // C string is terminated with a zero
    do str.with_c_str |buf| {
        unsafe {
            libc::my_strlen(buf) as uint
        }
    }
}

pub fn main() {
    let len = strlen(~"Rust");
    assert_eq!(len, 4u);
}
