// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core,unboxed_closures)]

use std::marker::PhantomData;

// Test that we are able to infer a suitable kind for a "recursive"
// closure.  As far as I can tell, coding up a recursive closure
// requires the good ol' [Y Combinator].
//
// [Y Combinator]: http://en.wikipedia.org/wiki/Fixed-point_combinator#Y_combinator

struct YCombinator<F,A,R> {
    func: F,
    marker: PhantomData<(A,R)>,
}

impl<F,A,R> YCombinator<F,A,R> {
    fn new(f: F) -> YCombinator<F,A,R> {
        YCombinator { func: f, marker: PhantomData }
    }
}

impl<A,R,F : Fn(&Fn(A) -> R, A) -> R> Fn<(A,)> for YCombinator<F,A,R> {
    type Output = R;

    extern "rust-call" fn call(&self, (arg,): (A,)) -> R {
        (self.func)(self, arg)
    }
}

fn main() {
    let factorial = |recur: &Fn(u32) -> u32, arg: u32| -> u32 {
        if arg == 0 {1} else {arg * recur(arg-1)}
    };
    let factorial: YCombinator<_,u32,u32> = YCombinator::new(factorial);
    let r = factorial(10);
    assert_eq!(3628800, r);
}
