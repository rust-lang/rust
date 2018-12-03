// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #7526: lowercase static constants in patterns look like bindings

#![deny(misleading_constant_patterns)]
#![allow(dead_code)]

#[allow(non_upper_case_globals)]
pub const a : isize = 97;

fn f() {
    let r = match (0,0) {
        (0, a) => 0, //~ ERROR should be upper case
                     //| HELP convert the pattern to upper case
                     //| SUGGESTION A
        (x, y) => 1 + x + y,
    };
    assert_eq!(r, 1);
}

mod m {
    #[allow(non_upper_case_globals)]
    pub const aha : isize = 7;
}

fn g() {
    use self::m::aha;
    let r = match (0,0) {
        (0, aha) => 0, //~ ERROR should be upper case
                       //| HELP convert the pattern to upper case
                       //| SUGGESTION AHA
        (x, y)   => 1 + x + y,
    };
    assert_eq!(r, 1);
}

mod n {
    pub const OKAY : isize = 8;
}

fn h() {
    use self::n::OKAY as not_okay;
    let r = match (0,0) {
        (0, not_okay) => 0, //~ ERROR should be upper case
                            //| HELP convert the pattern to upper case
                            //| SUGGESTION NOT_OKAY
        (x, y)   => 1 + x + y,
    };
    assert_eq!(r, 1);
}

pub const A : isize = 97;

fn i() {
    let r = match (0,0) {
        (0, A) => 0,
        (x, y) => 1 + x + y,
    };
    assert_eq!(r, 1);
    let r = match (0,97) {
        (0, A) => 0,
        (x, y) => 1 + x + y,
    };
    assert_eq!(r, 0);
}

fn j() {
    use self::m::aha as AHA;
    let r = match (0,0) {
        (0, AHA) => 0,
        (x, y)   => 1 + x + y,
    };
    assert_eq!(r, 1);
    let r = match (0,7) {
        (0, AHA) => 0,
        (x, y)   => 1 + x + y,
    };
    assert_eq!(r, 0);
}

fn k() {
    let r = match (0,0) {
        (0, self::m::aha) => 0,
        (x, y)      => 1 + x + y,
    };
    assert_eq!(r, 1);
    let r = match (0,7) {
        (0, self::m::aha) => 0,
        (x, y)      => 1 + x + y,
    };
    assert_eq!(r, 0);
}

fn main() {}
