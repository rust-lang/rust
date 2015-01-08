// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME #20661: format_args! emits calls to the unstable std::fmt::rt
// module, so the compiler has some hacks to make that possible
// (in span_is_internal). Unnfortunately those hacks defeat this
// particular scenario of checking feature gates in arguments to
// println!().

// ignore-test

// tests that input to a macro is checked for use of gated features. If this
// test succeeds due to the acceptance of a feature, pick a new feature to
// test. Not ideal, but oh well :(

fn main() {
    let a = &[1i32, 2, 3];
    println!("{}", {
        extern "rust-intrinsic" { //~ ERROR intrinsics are subject to change
            fn atomic_fence();
        }
        atomic_fence();
        42
    });
}
