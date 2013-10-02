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

#[deny(non_uppercase_pattern_statics)];

pub static a : int = 97;

fn f() {
    let r = match (0,0) {
        (0, a) => 0,
        //~^ ERROR static constant in pattern should be all caps
        (x, y) => 1 + x + y,
    };
    assert!(r == 1);
}

mod m {
    pub static aha : int = 7;
}

fn g() {
    use self::m::aha;
    let r = match (0,0) {
        (0, aha) => 0,
        //~^ ERROR static constant in pattern should be all caps
        (x, y)   => 1 + x + y,
    };
    assert!(r == 1);
}

mod n {
    pub static OKAY : int = 8;
}

fn h() {
    use not_okay = self::n::OKAY;
    let r = match (0,0) {
        (0, not_okay) => 0,
        //~^ ERROR static constant in pattern should be all caps
        (x, y)   => 1 + x + y,
    };
    assert!(r == 1);
}

fn main () {
    f();
    g();
    h();
}
