// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(never_type)]
#![allow(unreachable_code)]

fn main() {
    // The `if false` expressions are simply to
    // make sure we don't avoid checking everything
    // simply because a few expressions are unreachable.

    if false {
        let _: ! = {
            loop { return } // ok
        };
    }

    if false {
        let _: ! = {
            loop { return; break } // ok
        };
    }

    if false {
        let _: ! = {
            // Here, the break (implicitly carrying the value `()`)
            // occurs before the return, so it doesn't have the type
            // `!` and should thus fail to type check.
            loop { return break } //~ ERROR mismatched types
        };
    }

    if false {
        let _: ! = {
            loop { break } //~ ERROR mismatched types
        };
    }
}
