// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=compare

fn ok() {
    loop {
        let _x = 1;
    }
}

fn also_ok() {
    loop {
        let _x = String::new();
    }
}

fn fail() {
    loop {
        let x: i32;
        let _ = x + 1; //~ERROR (Ast) [E0381]
                       //~^ ERROR (Mir) [E0381]
    }
}

fn main() {
    ok();
    also_ok();
    fail();
}
