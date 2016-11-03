// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--test
// rustc-env:RUSTC_BOOTSTRAP_KEY=
#![cfg(any())] // This test should be configured away
#![feature(rustc_attrs)] // Test that this is allowed on stable/beta
#![feature(iter_arith_traits)] // Test that this is not unused
#![deny(unused_features)]

#[test]
fn dummy() {
    let () = "this should not reach type-checking";
}
