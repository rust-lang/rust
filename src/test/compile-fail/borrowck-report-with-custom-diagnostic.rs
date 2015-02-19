// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
fn main() {
    // Original borrow ends at end of function
    let mut x = 1_usize;
    let y = &mut x;
    let z = &x; //~ ERROR cannot borrow
}
//~^ NOTE previous borrow ends here

fn foo() {
    match true {
        true => {
            // Original borrow ends at end of match arm
            let mut x = 1_usize;
            let y = &x;
            let z = &mut x; //~ ERROR cannot borrow
        }
     //~^ NOTE previous borrow ends here
        false => ()
    }
}

fn bar() {
    // Original borrow ends at end of closure
    || {
        let mut x = 1_usize;
        let y = &mut x;
        let z = &mut x; //~ ERROR cannot borrow
    };
 //~^ NOTE previous borrow ends here
}
