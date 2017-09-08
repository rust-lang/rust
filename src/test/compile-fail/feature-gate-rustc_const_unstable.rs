// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test internal const fn feature gate.

#![feature(staged_api)]
#![feature(const_fn)]
//#![feature(rustc_const_unstable)]

#[stable(feature="zing", since="1.0.0")]
#[rustc_const_unstable(feature="fzzzzzt")] //~ERROR internal feature
pub const fn bazinga() {}

fn main() {
}

