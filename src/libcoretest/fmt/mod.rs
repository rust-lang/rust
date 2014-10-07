// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::fmt::Show;

mod num;

#[test]
fn test_format_flags() {
    // No residual flags left by pointer formatting
    let p = "".as_ptr();
    assert_eq!(format!("{:p} {:x}", p, 16u), format!("{:p} 10", p));
}

#[test]
fn test_len_hints_float() {
    let mut f = 5.0f32;
    for _ in range(0u, 30) {
        let s = format!("{}", f);
        let len = f.formatter_len_hint().unwrap();
        assert!(len >= s.len());
        assert!(len <= 128);
        f /= 10.0;
    }
    let mut f = 5.0f32;
    for _ in range(0u, 30) {
        let s = format!("{}", f);
        let len = f.formatter_len_hint().unwrap();
        assert!(len >= s.len());
        assert!(len <= 128);
        f *= 10.0;
    }
}

#[test]
fn test_len_hints_u64() {
    let mut f = 1u64;
    for _ in range(0u, 20) {
        let s = format!("{}", f);
        let len = f.formatter_len_hint().unwrap();
        assert!(len >= s.len());
        assert!(len <= 128);
        f *= 10;
    }
}
