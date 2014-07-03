// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// We only want to assert that this doesn't ICE, we don't particularly care
// about whether it nor it fails to compile.

// error-pattern:

#![feature(macro_rules)]

macro_rules! foo{
    () => {{
        macro_rules! bar{() => (())}
        1
    }}
}

pub fn main() {
    foo!();

    assert!({one! two()});

    // regardless of whether nested macro_rules works, the following should at
    // least throw a conventional error.
    assert!({one! two});
}

