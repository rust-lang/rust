// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "cdylib"]
#![deny(warnings)]

#[link_section = "foo"]
pub static A: [u8; 2] = [1, 2];

// make sure this is in another CGU
pub mod another {
    #[link_section = "foo"]
    pub static FOO: [u8; 2] = [3, 4];

    pub fn foo() {}
}

#[no_mangle]
pub extern fn foo() {
    // This will import `another::foo` through ThinLTO passes, and it better not
    // also accidentally import the `FOO` custom section into this module as
    // well
    another::foo();
}
