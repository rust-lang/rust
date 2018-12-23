// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// compile-flags:-Zborrowck=compare

#![allow(warnings)]
#![feature(rustc_attrs)]


fn main() {
}

fn nll_fail() {
    let mut data = ('a', 'b', 'c');
    let c = &mut data.0;
    capitalize(c);
    data.0 = 'e';
    //~^ ERROR (Ast) [E0506]
    //~| ERROR (Mir) [E0506]
    data.0 = 'f';
    //~^ ERROR (Ast) [E0506]
    data.0 = 'g';
    //~^ ERROR (Ast) [E0506]
    capitalize(c);
}

fn nll_ok() {
    let mut data = ('a', 'b', 'c');
    let c = &mut data.0;
    capitalize(c);
    data.0 = 'e';
    //~^ ERROR (Ast) [E0506]
    data.0 = 'f';
    //~^ ERROR (Ast) [E0506]
    data.0 = 'g';
    //~^ ERROR (Ast) [E0506]
}

fn capitalize(_: &mut char) {
}
