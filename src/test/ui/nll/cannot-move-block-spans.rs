// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the we point to the inner expression when moving out to initialize
// a variable, and that we don't give a useless suggestion such as &{ *r }.

pub fn deref(r: &String) {
    let x = { *r }; //~ ERROR
    let y = unsafe { *r }; //~ ERROR
    let z = loop { break *r; }; //~ ERROR
}

pub fn index(arr: [String; 2]) {
    let x = { arr[0] }; //~ ERROR
    let y = unsafe { arr[0] }; //~ ERROR
    let z = loop { break arr[0]; }; //~ ERROR
}

pub fn additional_statement_cases(r: &String) {
    let x = { let mut u = 0; u += 1; *r }; //~ ERROR
    let y = unsafe { let mut u = 0; u += 1; *r }; //~ ERROR
    let z = loop { let mut u = 0; u += 1; break *r; u += 2; }; //~ ERROR
}

fn main() {}
