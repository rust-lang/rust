// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const TEST_DATA: [u8; 32 * 1024 * 1024] = [42; 32 * 1024 * 1024];

// Check that the promoted copy of TEST_DATA doesn't
// leave an alloca from an unused temp behind, which,
// without optimizations, can still blow the stack.
fn main() {
    println!("{}", TEST_DATA.len());
}
