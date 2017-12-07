// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic test for free regions in the NLL code. This test does not
// report an error because of the (implied) bound that `'b: 'a`.

// compile-flags:-Znll -Zborrowck=mir -Zverbose
// must-compile-successfully

#![allow(warnings)]

fn foo<'a, 'b>(x: &'a &'b u32) -> &'a u32 {
    &**x
}

fn main() { }
