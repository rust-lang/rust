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
// lifetimes.
//
// Currently, we accept this code (with NLL enabled) since `x` is only
// borrowed once at a time -- though whether we should is not entirely
// obvious to me (the borrows are live over a yield, but then they are
// re-borrowing borrowed content, etc). Maybe I just haven't had
// enough coffee today, but I'm not entirely sure at this moment what
// effect a `suspend` should have on existing borrows. -nmatsakis

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
