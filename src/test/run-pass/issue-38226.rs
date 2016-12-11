// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test makes sure that we don't run into a linker error because of the
// middle::reachable pass missing trait methods with default impls.

// aux-build:issue_38226_aux.rs

// Need -Cno-prepopulate-passes to really disable inlining, otherwise the faulty
// code gets optimized out:
// compile-flags: -Cno-prepopulate-passes

extern crate issue_38226_aux;

fn main() {
    issue_38226_aux::foo::<()>();
}
