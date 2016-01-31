// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Test that Clone and other common traits are object-safe.

#![feature(zero_one)]

macro_rules! check {
    ($($trate:path),*) => {
        $(let _: Box<$trate>;)*
    }
}

fn main() {
    check!(Clone,
           Default,
           From<()>,
           Into<()>,
           std::str::FromStr<Err=()>,
           std::num::One,
           std::num::Zero);
}

