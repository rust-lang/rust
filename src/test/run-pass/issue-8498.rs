// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

pub fn main() {
// This is ok
    match &[(~5,~7)] {
        ps => {
           let (ref y, _) = ps[0];
           println(fmt!("1. y = %d", **y));
           assert!(**y == 5);
        }
    }

// This is not entirely ok
    match Some(&[(~5,)]) {
        Some(ps) => {
           let (ref y,) = ps[0];
           println(fmt!("2. y = %d", **y));
           if **y != 5 { println("sadness"); }
        }
        None => ()
    }

// This is not ok
    match Some(&[(~5,~7)]) {
        Some(ps) => {
           let (ref y, ref z) = ps[0];
           println(fmt!("3. y = %d z = %d", **y, **z));
           assert!(**y == 5);
        }
        None => ()
    }
}

