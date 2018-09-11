// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:llvm_pr32379.rs

// LLVM PR #32379 (https://bugs.llvm.org/show_bug.cgi?id=32379), which
// applies to upstream LLVM 3.9.1, is known to cause rustc itself to be
// miscompiled on ARM (Rust issue #40593). Because cross builds don't test
// our *compiler* on ARM, have a test for the miscompilation directly.

extern crate llvm_pr32379;

pub fn main() {
    let val = llvm_pr32379::pr32379(2, false, false);
    assert_eq!(val, 2);
}
