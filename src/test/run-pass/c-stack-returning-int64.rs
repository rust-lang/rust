// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate libc;

mod mlibc {
    use libc::{c_char, c_long, c_longlong};

    extern {
        pub fn atol(x: *c_char) -> c_long;
        pub fn atoll(x: *c_char) -> c_longlong;
    }
}

fn atol(s: ~str) -> int {
    s.with_c_str(|x| unsafe { mlibc::atol(x) as int })
}

fn atoll(s: ~str) -> i64 {
    s.with_c_str(|x| unsafe { mlibc::atoll(x) as i64 })
}

pub fn main() {
    assert_eq!(atol(~"1024") * 10, atol(~"10240"));
    assert!((atoll(~"11111111111111111") * 10) == atoll(~"111111111111111110"));
}
