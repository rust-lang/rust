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

fn main() {
    // The `if false` expressions are simply to
    // make sure we don't avoid checking everything
    // simply because a few expressions are unreachable.

    if false {
        let _: ! = { //~ ERROR mismatched types
            'a: while break 'a {};
        };
    }

    if false {
        let _: ! = {
            while false { //~ ERROR mismatched types
                break
            }
        };
    }

    if false {
        let _: ! = {
            while false { //~ ERROR mismatched types
                return
            }
        };
    }
}
