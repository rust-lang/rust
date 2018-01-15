// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn bar<F: Fn()>(_f: F) {}

pub fn foo() {
    let mut x = 0;
    bar(move || x = 1);
    //~^ ERROR cannot assign to captured outer variable in an `Fn` closure
    //~| NOTE `Fn` closures cannot capture their enclosing environment for modifications
}

fn main() {}
