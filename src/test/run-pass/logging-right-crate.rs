// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:logging_right_crate.rs
// xfail-fast
// exec-env:RUST_LOG=logging-right-crate=debug

// This is a test for issue #3046 to make sure that when we monomorphize a
// function from one crate to another the right top-level logging name is
// preserved.
//
// It used to be the case that if logging were turned on for this crate, all
// monomorphized functions from other crates had logging turned on (their
// logging module names were all incorrect). This test ensures that this no
// longer happens by enabling logging for *this* crate and then invoking a
// function in an external crate which will fail when logging is enabled.

extern mod logging_right_crate;

pub fn main() {
    // this function fails if logging is turned on
    logging_right_crate::foo::<int>();
}
