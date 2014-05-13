// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast doesn't like extern crate

extern crate libc;

mod mlibc {
    extern crate libc;
    use self::libc::{c_char, size_t};

    extern {
        #[link_name = "strlen"]
        pub fn my_strlen(str: *c_char) -> size_t;
    }
}

fn strlen(str: StrBuf) -> uint {
    // C string is terminated with a zero
    str.as_slice().with_c_str(|buf| {
        unsafe {
            mlibc::my_strlen(buf) as uint
        }
    })
}

pub fn main() {
    let len = strlen("Rust".to_strbuf());
    assert_eq!(len, 4u);
}
