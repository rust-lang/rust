// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that declarations and types which use `extern fn` both have the same
// ABI (#9309).

extern {
    fn printf();
}

pub fn main() {
    // Will only type check if the type of _p and the decl of printf use the same ABI
    let _p: extern unsafe fn() = printf;
}
