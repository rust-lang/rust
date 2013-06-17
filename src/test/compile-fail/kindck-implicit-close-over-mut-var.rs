// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

fn user(_i: int) {}

fn foo() {
    // Here, i is *moved* into the closure: Not actually OK
    let mut i = 0;
    do task::spawn {
        user(i); //~ ERROR mutable variables cannot be implicitly captured
    }
}

fn bar() {
    // Here, i would be implicitly *copied* but it
    // is mutable: bad
    let mut i = 0;
    while i < 10 {
        do task::spawn {
            user(i); //~ ERROR mutable variables cannot be implicitly captured
        }
        i += 1;
    }
}

fn car() {
    // Here, i is mutable, but *explicitly* shadowed copied:
    let mut i = 0;
    while i < 10 {
        {
            let i = i;
            do task::spawn {
                user(i);
            }
        }
        i += 1;
    }
}

fn main() {
}
