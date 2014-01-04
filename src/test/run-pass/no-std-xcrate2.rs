// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test: this has weird linking problems on linux, and it probably needs a
//             solution along the lines of disabling segmented stacks and/or the
//             stack checks.
// aux-build:no_std_crate.rs

// This tests that libraries built with #[no_std] can be linked to crates with
// #[no_std] and actually run.

#[no_std];

extern mod no_std_crate;

// This is an unfortunate thing to have to do on linux :(
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub mod linkhack {
    #[link_args="-lrustrt -lrt"]
    extern {}
}

#[start]
pub fn main(_: int, _: **u8, _: *u8) -> int {
    no_std_crate::foo();
    0
}
