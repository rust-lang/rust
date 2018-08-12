// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Convert unsigned integers into a string representation with some base.
/// Bases up to and including 36 can be used for case-insensitive things.

use std::str;

pub const MAX_BASE: usize = 64;
pub const ALPHANUMERIC_ONLY: usize = 62;
pub const CASE_INSENSITIVE: usize = 36;

const BASE_64: &[u8; MAX_BASE as usize] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@$";

#[inline]
pub fn push_str(mut n: u128, base: usize, output: &mut String) {
    debug_assert!(base >= 2 && base <= MAX_BASE);
    let mut s = [0u8; 128];
    let mut index = 0;

    let base = base as u128;

    loop {
        s[index] = BASE_64[(n % base) as usize];
        index += 1;
        n /= base;

        if n == 0 {
            break;
        }
    }
    s[0..index].reverse();

    output.push_str(str::from_utf8(&s[0..index]).unwrap());
}

#[inline]
pub fn encode(n: u128, base: usize) -> String {
    let mut s = String::new();
    push_str(n, base, &mut s);
    s
}

#[test]
fn test_encode() {
    fn test(n: u128, base: usize) {
        assert_eq!(Ok(n), u128::from_str_radix(&encode(n, base), base as u32));
    }

    for base in 2..37 {
        test(0, base);
        test(1, base);
        test(35, base);
        test(36, base);
        test(37, base);
        test(u64::max_value() as u128, base);
        test(u128::max_value(), base);

        for i in 0 .. 1_000 {
            test(i * 983, base);
        }
    }
}
