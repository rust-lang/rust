// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that, when a variable of type `&T` is captured inside a proc,
// we correctly infer/require that its lifetime is 'static.

fn foo(_p: proc():'static) { }

static i: int = 3;

fn capture_local() {
    let x = 3i;
    let y = &x; //~ ERROR `x` does not live long enough
    foo(proc() {
        let _a = *y;
    });
}

fn capture_static() {
    // Legal because &i can have static lifetime:
    let y = &i;
    foo(proc() {
        let _a = *y;
    });
}

fn main() { }
