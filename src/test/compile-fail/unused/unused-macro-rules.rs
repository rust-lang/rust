// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_macros)]

// Most simple case
macro_rules! unused { //~ ERROR: unused macro definition
    () => {};
}

// Test macros created by macros
macro_rules! create_macro {
    () => {
        macro_rules! m { //~ ERROR: unused macro definition
            () => {};
        }
    };
}
create_macro!();

#[allow(unused_macros)]
mod bar {
    // Test that putting the #[deny] close to the macro's definition
    // works.

    #[deny(unused_macros)]
    macro_rules! unused { //~ ERROR: unused macro definition
        () => {};
    }
}

fn main() {}
