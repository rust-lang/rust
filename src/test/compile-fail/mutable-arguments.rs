// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Note: it would be nice to give fewer warnings in these cases.

fn mutate_by_mut_ref(x: &mut uint) {
    *x = 0;
}

fn mutate_by_ref(&&x: uint) {
    //~^ WARNING unused variable: `x`
    x = 0; //~ ERROR assigning to argument
}

fn mutate_by_val(++x: uint) {
    //~^ WARNING unused variable: `x`
    x = 0; //~ ERROR assigning to argument
}

fn mutate_by_copy(+x: uint) {
    //~^ WARNING unused variable: `x`
    x = 0; //~ ERROR assigning to argument
    //~^ WARNING value assigned to `x` is never read
}

fn mutate_by_move(-x: uint) {
    //~^ WARNING unused variable: `x`
    x = 0; //~ ERROR assigning to argument
    //~^ WARNING value assigned to `x` is never read
}

fn main() {
}
