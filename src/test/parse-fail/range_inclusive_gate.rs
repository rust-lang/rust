// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that #![feature(inclusive_range_syntax)] is required.

// #![feature(inclusive_range_syntax, inclusive_range)]

macro_rules! m {
    () => { for _ in 1...10 {} } //~ ERROR inclusive range syntax is experimental
}

#[cfg(nope)]
fn f() {}
#[cfg(not(nope))]
fn f() {
    for _ in 1...10 {} //~ ERROR inclusive range syntax is experimental
}

#[cfg(nope)]
macro_rules! n { () => {} }
#[cfg(not(nope))]
macro_rules! n {
    () => { for _ in 1...10 {} } //~ ERROR inclusive range syntax is experimental
}

macro_rules! o {
    () => {{
        #[cfg(nope)]
        fn g() {}
        #[cfg(not(nope))]
        fn g() {
            for _ in 1...10 {} //~ ERROR inclusive range syntax is experimental
        }

        g();
    }}
}

#[cfg(nope)]
macro_rules! p { () => {} }
#[cfg(not(nope))]
macro_rules! p {
    () => {{
        #[cfg(nope)]
        fn h() {}
        #[cfg(not(nope))]
        fn h() {
            for _ in 1...10 {} //~ ERROR inclusive range syntax is experimental
        }

        h();
    }}
}

pub fn main() {
    for _ in 1...10 {} //~ ERROR inclusive range syntax is experimental
    for _ in ...10 {} //~ ERROR inclusive range syntax is experimental

    f(); // not allowed in cfg'ed functions

    m!(); // not allowed in macros
    n!(); // not allowed in cfg'ed macros
    o!(); // not allowed in macros that output cfgs
    p!(); // not allowed in cfg'ed macros that output cfgs
}


