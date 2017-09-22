// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --cap-lints allow

// This tests that the fn_must_use feature-gate warning respects the lint
// cap. (See discussion in Issue #44213.)

#![feature(rustc_attrs)]

#[must_use] // (no feature-gate warning because of the lint cap!)
fn need_to_use_it() -> bool { true }

#[rustc_error]
fn main() {} //~ ERROR compilation successful
