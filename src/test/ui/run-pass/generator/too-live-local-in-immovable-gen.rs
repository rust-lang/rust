// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators)]

fn main() {
    unsafe {
        static move || {
            // Tests that the generator transformation finds out that `a` is not live
            // during the yield expression. Type checking will also compute liveness
            // and it should also find out that `a` is not live.
            // The compiler will panic if the generator transformation finds that
            // `a` is live and type checking finds it dead.
            let a = {
                yield ();
                4i32
            };
            &a;
        };
    }
}
