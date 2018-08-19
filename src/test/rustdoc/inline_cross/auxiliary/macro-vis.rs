// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "qwop"]

/// (writen on a spider's web) Some Macro
#[macro_export]
macro_rules! some_macro {
    () => {
        println!("this is some macro, for sure");
    };
}

/// Some other macro, to fill space.
#[macro_export]
macro_rules! other_macro {
    () => {
        println!("this is some other macro, whatev");
    };
}

/// This macro is so cool, it's Super.
#[macro_export]
macro_rules! super_macro {
    () => {
        println!("is it a bird? a plane? no, it's Super Macro!");
    };
}
