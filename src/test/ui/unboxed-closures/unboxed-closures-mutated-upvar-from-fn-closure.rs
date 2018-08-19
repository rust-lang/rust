// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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

// Test that a by-ref `FnMut` closure gets an error when it tries to
// mutate a value.

fn call<F>(f: F) where F : Fn() {
    f();
}

fn main() {
    let mut counter = 0;
    call(|| {
        counter += 1;
        //[ast]~^ ERROR cannot assign to data in a captured outer variable in an `Fn` closure
        //[mir]~^^ ERROR cannot assign to `counter`
    });
}
