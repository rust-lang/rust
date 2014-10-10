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

// This is similar to compile-fail/match-static-const-lc, except it
// shows the expected usual workaround (choosing a different name for
// the static definition) and also demonstrates that one can work
// around this problem locally by renaming the constant in the `use`
// form to an uppercase identifier that placates the lint.

#![deny(non_uppercase_statics)]

pub const A : int = 97;

fn f() {
    let r = match (0,0) {
        (0, A) => 0,
        (x, y) => 1 + x + y,
    };
    assert!(r == 1);
    let r = match (0,97) {
        (0, A) => 0,
        (x, y) => 1 + x + y,
    };
    assert!(r == 0);
}

mod m {
    #[allow(non_uppercase_statics)]
    pub const aha : int = 7;
}

fn g() {
    use self::m::aha as AHA;
    let r = match (0,0) {
        (0, AHA) => 0,
        (x, y)   => 1 + x + y,
    };
    assert!(r == 1);
    let r = match (0,7) {
        (0, AHA) => 0,
        (x, y)   => 1 + x + y,
    };
    assert!(r == 0);
}

fn h() {
    let r = match (0,0) {
        (0, self::m::aha) => 0,
        (x, y)      => 1 + x + y,
    };
    assert!(r == 1);
    let r = match (0,7) {
        (0, self::m::aha) => 0,
        (x, y)      => 1 + x + y,
    };
    assert!(r == 0);
}

pub fn main () {
    f();
    g();
    h();
}
