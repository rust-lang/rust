// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test case tests whether we can handle code bases that contain a high
// number of closures, something that needs special handling in the MingGW
// toolchain.
// See https://github.com/rust-lang/rust/issues/34793 for more information.

// Make sure we don't optimize anything away:
// compile-flags: -C no-prepopulate-passes

// Expand something exponentially
macro_rules! go_bacterial {
    ($mac:ident) => ($mac!());
    ($mac:ident 1 $($t:tt)*) => (
        go_bacterial!($mac $($t)*);
        go_bacterial!($mac $($t)*);
    )
}

macro_rules! mk_closure {
    () => ((move || {})())
}

macro_rules! mk_fn {
    () => {
        {
            fn function() {
                // Make 16 closures
                go_bacterial!(mk_closure 1 1 1 1);
            }
            let _ = function();
        }
    }
}

fn main() {
    // Make 2^12 functions, each containing 16 closures,
    // resulting in 2^16 closures overall.
    go_bacterial!(mk_fn 1 1 1 1  1 1 1 1  1 1 1 1);
}
