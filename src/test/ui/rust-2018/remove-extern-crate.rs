// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-rustfix
// edition:2018
// compile-pass
// aux-build:remove-extern-crate.rs
// compile-flags:--extern remove_extern_crate

#![warn(rust_2018_idioms)]

extern crate core;
extern crate core as another_name;
use remove_extern_crate;
#[macro_use]
extern crate remove_extern_crate as something_else;

fn main() {
    another_name::mem::drop(3);
    another::foo();
    remove_extern_crate::foo!();
    bar!();
}

mod another {
    extern crate core;
    use remove_extern_crate;

    pub fn foo() {
        core::mem::drop(4);
        remove_extern_crate::foo!();
    }
}
