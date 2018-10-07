// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![allow(unreachable_patterns)]
#![allow(unused_variables)]
#![warn(unused_parens)]

fn main() {
    match 1 {
        (_) => {}         //~ WARNING: unnecessary parentheses around pattern
        (y) => {}         //~ WARNING: unnecessary parentheses around pattern
        (ref r) => {}     //~ WARNING: unnecessary parentheses around pattern
        (e @ 1..=2) => {} //~ WARNING: unnecessary parentheses around outer pattern
        (1..=2) => {}     // Non ambiguous range pattern should not warn
        e @ (3..=4) => {} // Non ambiguous range pattern should not warn
    }

    match &1 {
        (e @ &(1...2)) => {} //~ WARNING: unnecessary parentheses around outer pattern
        &(_) => {}           //~ WARNING: unnecessary parentheses around pattern
        e @ &(1...2) => {}   // Ambiguous range pattern should not warn
        &(1..=2) => {}       // Ambiguous range pattern should not warn
    }

    match &1 {
        e @ &(1...2) | e @ &(3..=4) => {} // Complex ambiguous pattern should not warn
        &_ => {}
    }
}
