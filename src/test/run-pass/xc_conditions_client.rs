// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// aux-build:xc_conditions.rs

extern mod xc_conditions;
use xc_conditions::oops;
use xc_conditions::trouble;

// Tests of cross-crate conditions; the condition is
// defined in lib, and we test various combinations
// of `trap` and `raise` in the client or the lib where
// the condition was defined. Also in test #4 we use
// more complex features (generics, traits) in
// combination with the condition.
//
//                    trap   raise
//                    ------------
// xc_conditions  :   client   lib
// xc_conditions_2:   client   client
// xc_conditions_3:   lib      client
// xc_conditions_4:   client   client  (with traits)
//
// the trap=lib, raise=lib case isn't tested since
// there's no cross-crate-ness to test in that case.

pub fn main() {
    do oops::cond.trap(|_i| 12345).inside {
        let x = trouble();
        assert_eq!(x,12345);
    }
}
