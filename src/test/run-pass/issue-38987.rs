// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(i128_type)]

// SNAP: run on all stages after snapshot, i128 currently doesn't work on stages 0 and 1
// ignore-stage1
// ignore-stage0

fn main() {
    let _ = -0x8000_0000_0000_0000_0000_0000_0000_0000i128;
}
