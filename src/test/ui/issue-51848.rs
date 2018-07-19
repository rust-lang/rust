// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// In case of macro expansion, the errors should be matched using the deepest callsite in the
// macro call stack whose span is in the current file

macro_rules! macro_with_error {
    ( ) => {
        println!("{"); //~ ERROR invalid
    };
}

fn foo() {

}

fn main() {
    macro_with_error!();
    //^ In case of a local macro we want the error to be matched in the macro definition, not here

    println!("}"); //~ ERROR invalid
    //^ In case of an external macro we want the error to be matched here
}
