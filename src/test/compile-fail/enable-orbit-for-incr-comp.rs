// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty
// compile-flags:-Zincremental=tmp/cfail-tests/enable-orbit-for-incr-comp -Zorbit=off
// error-pattern:Automatically enabling `-Z orbit` because `-Z incremental` was specified

#![deny(warnings)]

fn main() {
    FAIL! // We just need some compilation error. What we really care about is
          // that the error pattern above is checked.
}
