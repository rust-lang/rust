// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// tests that input to a macro is checked for use of gated features. If this
// test succeeds due to the acceptance of a feature, pick a new feature to
// test. Not ideal, but oh well :(

fn main() {
    let a = &[1, 2, 3];
    println!("{}", {
        extern "rust-intrinsic" { //~ ERROR intrinsics are subject to change
            fn atomic_fence();
        }
        atomic_fence();
        42
    });
}
