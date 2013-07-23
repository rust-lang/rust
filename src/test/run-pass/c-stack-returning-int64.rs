// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::str;

mod libc {
    #[abi = "cdecl"]
    #[nolink]
    extern {
        pub fn atol(x: *u8) -> int;
        pub fn atoll(x: *u8) -> i64;
    }
}

fn atol(s: ~str) -> int {
    s.as_imm_buf(|x, _len| unsafe { libc::atol(x) })
}

fn atoll(s: ~str) -> i64 {
    s.as_imm_buf(|x, _len| unsafe { libc::atoll(x) })
}

pub fn main() {
    unsafe {
        assert_eq!(atol(~"1024") * 10, atol(~"10240"));
        assert!((atoll(~"11111111111111111") * 10i64)
            == atoll(~"111111111111111110"));
    }
}
