// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:impl_privacy_xc_2.rs
// ignore-fast

extern crate impl_privacy_xc_2;

pub fn main() {
    let fish1 = impl_privacy_xc_2::Fish { x: 1 };
    let fish2 = impl_privacy_xc_2::Fish { x: 2 };
    if fish1.eq(&fish2) { println!("yes") } else { println!("no") };
}
