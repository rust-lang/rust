// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks if an unstable feature is enabled with the -Zcrate-attr=feature(foo) flag. If
// the exact feature used here is causing problems feel free to replace it with another
// perma-unstable feature.

// compile-flags: -Zcrate-attr=feature(abi_unadjusted)

#![allow(dead_code)]

extern "unadjusted" fn foo() {}

fn main() {}
