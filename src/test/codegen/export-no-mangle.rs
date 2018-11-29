// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

mod private {
    // CHECK: @FOO =
    #[no_mangle]
    pub static FOO: u32 = 3;

    // CHECK: @BAR =
    #[export_name = "BAR"]
    static BAR: u32 = 3;

    // CHECK: void @foo()
    #[no_mangle]
    pub extern fn foo() {}

    // CHECK: void @bar()
    #[export_name = "bar"]
    extern fn bar() {}
}
