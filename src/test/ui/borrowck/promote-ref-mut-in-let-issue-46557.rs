// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we fail to promote the constant here which has a `ref
// mut` borrow.

fn gimme_static_mut_let() -> &'static mut u32 {
    let ref mut x = 1234543; //~ ERROR
    x
}

fn gimme_static_mut_let_nested() -> &'static mut u32 {
    let (ref mut x, ) = (1234543, ); //~ ERROR
    x
}

fn gimme_static_mut_match() -> &'static mut u32 {
    match 1234543 {
        ref mut x => x //~ ERROR
    }
}

fn gimme_static_mut_match_nested() -> &'static mut u32 {
    match (123443,) {
        (ref mut x,) => x, //~ ERROR
    }
}

fn gimme_static_mut_ampersand() -> &'static mut u32 {
    &mut 1234543 //~ ERROR
}

fn main() {
}
