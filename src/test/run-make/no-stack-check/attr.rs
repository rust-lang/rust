// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]

extern {
    // Prevents optimizing away the stack buffer.
    // This symbol is undefined, but the code doesn't need to pass
    // the linker.
    fn black_box(ptr: *const u8);
}

#[no_stack_check]
pub unsafe fn foo() {
    // Make sure we use the stack
    let x: [u8, ..50] = [0, ..50];
    black_box(x.as_ptr());
}
