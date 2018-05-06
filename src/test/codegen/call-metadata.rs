// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that range metadata gets emitted on calls to functions returning a
// scalar value.

// compile-flags: -C no-prepopulate-passes
// min-llvm-version 4.0


#![crate_type = "lib"]

pub fn test() {
    // CHECK: call i8 @some_true(), !range [[R0:![0-9]+]]
    // CHECK: [[R0]] = !{i8 0, i8 3}
    some_true();
}

#[no_mangle]
fn some_true() -> Option<bool> {
    Some(true)
}
