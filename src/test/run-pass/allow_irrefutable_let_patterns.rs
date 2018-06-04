// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(irrefutable_let_pattern)]

// must-compile-successfully-irrefutable_let_pattern_with_gate
fn main() {
    #[allow(irrefutable_let_pattern)]
    if let _ = 5 {}

    #[allow(irrefutable_let_pattern)]
    while let _ = 5 {
        break;
    }
}
