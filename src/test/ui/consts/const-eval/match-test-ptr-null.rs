// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    // Make sure match uses the usual pointer comparison code path -- i.e., it should complain
    // that pointer comparison is disallowed, not that parts of a pointer are accessed as raw
    // bytes.
    let _: [u8; 0] = [4; { //~ ERROR could not evaluate repeat length
        match &1 as *const i32 as usize { //~ ERROR casting pointers to integers in constants
            0 => 42, //~ ERROR constant contains unimplemented expression type
            //~^ NOTE "pointer arithmetic or comparison" needs an rfc before being allowed
            n => n,
        }
    }];
}
