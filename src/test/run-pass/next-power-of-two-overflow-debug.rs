// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C debug_assertions=yes
// ignore-wasm32-bare compiled with panic=abort by default

#![feature(i128_type)]

use std::panic;

fn main() {
    macro_rules! overflow_test {
        ($t:ident) => (
            let r = panic::catch_unwind(|| {
                ($t::max_value()).next_power_of_two()
            });
            assert!(r.is_err());

            let r = panic::catch_unwind(|| {
                (($t::max_value() >> 1) + 2).next_power_of_two()
            });
            assert!(r.is_err());
        )
    }
    overflow_test!(u8);
    overflow_test!(u16);
    overflow_test!(u32);
    overflow_test!(u64);
    overflow_test!(u128);
}
