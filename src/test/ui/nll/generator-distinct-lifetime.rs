// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, nll)]

// Test for issue #47189. Here, both `s` and `t` are live for the
// generator's lifetime, but within the generator they have distinct
// lifetimes. We accept this code -- even though the borrow extends
// over a yield -- because the data that is borrowed (`*x`) is not
// stored on the stack.

// compile-pass

fn foo(x: &mut u32) {
    move || {
        let s = &mut *x;
        yield;
        *s += 1;

        let t = &mut *x;
        yield;
        *t += 1;
    };
}

fn main() {
    foo(&mut 0);
}
