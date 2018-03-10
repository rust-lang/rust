// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: Change to UI Test
// Check notes are placed on an assignment that can actually precede the current assigmnent
// Don't emmit a first assignment for assignment in a loop.

// compile-flags: -Zborrowck=compare

fn test() {
    let x;
    if true {
        x = 1;
    } else {
        x = 2;
        x = 3;      //~ ERROR (Ast) [E0384]
                    //~^ ERROR (Mir) [E0384]
    }
}

fn test_in_loop() {
    loop {
        let x;
        if true {
            x = 1;
        } else {
            x = 2;
            x = 3;      //~ ERROR (Ast) [E0384]
                        //~^ ERROR (Mir) [E0384]
        }
    }
}

fn test_using_loop() {
    let x;
    loop {
        if true {
            x = 1;      //~ ERROR (Ast) [E0384]
                        //~^ ERROR (Mir) [E0384]
        } else {
            x = 2;      //~ ERROR (Ast) [E0384]
                        //~^ ERROR (Mir) [E0384]
        }
    }
}

fn main() {}
