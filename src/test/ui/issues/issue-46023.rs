// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let x = 0;

    (move || {
        x = 1;
        //[mir]~^ ERROR cannot assign to `x`, as it is not declared as mutable [E0594]
        //[ast]~^^ ERROR cannot assign to captured outer variable in an `FnMut` closure [E0594]
    })()
}
