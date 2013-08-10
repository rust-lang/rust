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

use std::libc;
use std::str;
use std::vec;

mod libc {
    #[nolink]
    #[abi = "cdecl"]
    extern {
        #[link_name = "strlen"]
        pub fn my_strlen(str: *u8) -> uint;
    }
}

fn strlen(str: ~str) -> uint {
    unsafe {
        // C string is terminated with a zero
        do str.to_c_str().with_ref |buf| {
            libc::my_strlen(buf as *u8)
        }
    }
}

pub fn main() {
    let len = strlen(~"Rust");
    assert_eq!(len, 4u);
}
