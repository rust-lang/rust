// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #27300

fn main() {
    let _p = Some(45).and_then({|x|
        //~^ ERROR the trait bound `std::option::Option<_>: std::ops::FnOnce<({integer},)>`
        //~| NOTE the trait `std::ops::FnOnce<({integer},)>` is not implemented for
        println!("hi");
        Some(x * 2) //~ NOTE ...implicitly returns this
    });
    //~^^^^^^ WARN a closure's body is not determined by its enclosing block
    //~| NOTE this closure's body is not determined by its enclosing block
    //~| NOTE this is the closure's block...
    //~| NOTE ...while this enclosing block...
    //~| HELP you should open the block *after* the closure's argument list
    //~^^^^^^^ ERROR cannot find value `x` in this scope
    //~| NOTE not found in this scope

    // Don't alert on the folloing case, even though it is likely that the user is confused in the
    // same way as the first test case, as 1) clippy will warn about this and 2) if they ever
    // change it, the appropriate warning will trigger.
    let _y = Some(45).and_then({|x|
        Some(x * 2)
    });
}
